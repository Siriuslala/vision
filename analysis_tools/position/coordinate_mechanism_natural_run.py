"""Experiment 3.15 round 2 GPU stages (Qwen3.5-9B, bf16, SDPA, greedy, thinking off).

Results: docs/analysis/position/results_2026-11-05_coordinate_mechanism_natural.md.

Stages
  gen       direct point / bbox answers on the seven stimulus families (coordinate_mechanism_natural.FAMILIES)
            under three prefill conditions: clean, the Gated DeltaNet decay at image tokens of layers 9-16 set to
            alpha^s with s = 0 (the 3.15 graded-decay intervention), and 'delL' (the raster-preceding entity
            erased); row/column answers on grid10; text-only "centre of cell (r, c)" questions; and the same point
            / row-column questions with the located LoRA of experiment 3.13 r2 (layers 1-4, image tokens only).
  think     think-then-answer (CoT) generations on the RefCOCO crop windows A used by the flow stage.
  flow      activation patching per answer digit on (A, B) pairs that differ only in the target position
            (disc sweeps, grid10 neighbouring cells, RefCOCO crop-shift pairs for point, bbox and the first
            number written in a CoT). Every studied step is one generated digit; edits copy one class of run B's
            states into run A: image key/value at the studied step or over a 4-token window, previous-token
            Gated DeltaNet short-convolution slot, recurrent state, earlier answer-token key/value, or the output
            of chosen attention heads over the window; a scan makes B's image visible from answer position s on.
  timeline  teacher-forced answers; at every answer step: image attention share of every full-attention head,
            and the effect of hiding the image from that single step (all full-attention layers, layer 20 only,
            layers 16-24) or from every answer step so far; attention maps of layers 16-28 at each value's
            first-digit step (for the edge analysis).
  spot      batched vs single-sequence greedy decoding on 20 prompts.

Layers are 1-based; attention heads are 0-based indices into the 16 query heads (o_proj input slice h*256).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.behavior import cot_mechanism as cm
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_mechanism_natural as cmn
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import digit_failure_r2 as d2
from analysis_tools.position import edge_heads as eh
from analysis_tools.position import ordinal_vs_coordinate as ov
from analysis_tools.position import targeted_finetune as tf
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

FULL_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
LINEAR_LAYERS = tuple(l for l in range(1, 33) if l not in FULL_LAYERS)
GATE_LAYERS = tuple(range(9, 17))
HEAD_DIM = 256
MAP_LAYERS = (16, 20, 24, 28)
EDGE_LEFT = ((24, 7), (24, 1), (24, 2), (24, 12), (20, 0))          # 3.11 r2: left / top readers at point steps
EDGE_RIGHT = ((24, 5), (24, 6), (24, 8), (24, 9), (20, 10), (20, 13), (28, 0))
H11 = ((20, 11),)
R1_IMAGES = 'position/2026-10-16_readout_tool/images'
R2_CASES = 'position/2026-10-20_readout_tool_r2/cases.jsonl'
TFR2 = 'position/2026-11-03_targeted_finetune_r2'
LORA_COND = 'early1_4_img'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def code_digest() -> dict:
    here = Path(__file__).parent
    return {p: hashlib.sha256((here / p).read_bytes()).hexdigest()[:16]
            for p in ('coordinate_mechanism_natural.py', 'coordinate_mechanism_natural_run.py')}


# ----------------------------------------------------------------------------- cases and images

class Cases:
    """Builds the cases of every family and renders their images (thumbnail crops are loaded lazily)."""

    def __init__(self, args):
        self.args = args
        self._graphs = None
        self._pool = None

    def gqa_image(self, image_id: str) -> Image.Image:
        return Image.open(Path(self.args.data_dir) / 'GQA' / 'images' / 'images' / f'{image_id}.jpg').convert('RGB')

    @property
    def pool(self):
        if self._pool is None:
            g = json.loads((Path(self.args.data_dir) / 'GQA' / 'scene' / 'val_sceneGraphs.json').read_text())
            self._pool = cmn.thumb_pool(g)
        return self._pool

    def refcoco(self, n: int) -> list[dict]:
        cases = [c for c in read_jsonl(Path(self.args.work) / R2_CASES)
                 if c['set'] == 'refcoco' and c['split'] == 'test' and c['phrasing'] == 0]
        rng = np.random.default_rng(20261103)                 # the 3.15 round-1 selection
        pick = sorted(rng.choice(len(cases), min(n, len(cases)), replace=False))
        return [dict(cases[i], family='refcoco') for i in pick]

    def family(self, fam: str, n: int) -> list[dict]:
        if fam in ('grid10', 'grid8'):
            return cmn.grid_cases(fam, n)
        if fam == 'scatter':
            return cmn.scatter_cases(n)
        if fam == 'single':
            return cmn.single_cases(n)
        if fam in ('thumb_mixed', 'thumb_same'):
            return cmn.thumb_cases(self.pool, fam, n)
        if fam == 'refcoco':
            return self.refcoco(self.args.n_refcoco)
        raise ValueError(fam)

    def render(self, case: dict, variant: str = 'clean') -> tuple[Image.Image, dict]:
        fam = case['family']
        if fam in ('grid10', 'grid8'):
            n, cell = cmn.GRID_FAMILIES[fam]
            r, c = case['target']
            return cmn.synth_grid(n, cell, (r, c), case['template'], erase=(r, c - 1) if variant == 'delL' else None)
        if fam in ('thumb_mixed', 'thumb_same'):
            r, c = case['target']
            return cmn.thumb_image(case, self.gqa_image, erase=(r, c - 1) if variant == 'delL' else None)
        if fam == 'scatter':
            return cmn.scatter_image(case, erase_prev=variant == 'delL')
        if fam == 'single':
            return cmn.single_image(case)
        if fam == 'refcoco':
            img = Image.open(Path(self.args.work) / R1_IMAGES / case['image']).convert('RGB')
            t = case['truth']
            return img, cmn.truth_from_box(t['box'], tuple(t['canvas']))
        raise ValueError(fam)


# ----------------------------------------------------------------------------- gated generation

@contextmanager
def decay_scale(rt, s: float, image_mask: torch.Tensor):
    """alpha -> alpha^s at the image tokens (bool (B, L) of the prefill) in the DeltaNet layers 9-16 (prefill only)."""
    handles = []
    if s != 1.0:
        L = image_mask.shape[1]
        for layer in GATE_LAYERS:
            if layer in FULL_LAYERS:                       # layers 12 and 16 are full-attention layers
                continue
            mod = rt.layers[layer - 1].linear_attn

            def hook(m, inp, out, mod=mod):
                if out.shape[1] != L:
                    return out
                out = out.clone()
                msk = image_mask.to(out.device)
                out[msk] = ov.graded_a(out[msk], mod.dt_bias, s).to(out.dtype)
                return out
            handles.append(mod.in_proj_a.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def gen_batch(rt, items: list[tuple[Image.Image | None, str]], max_new: int, s: float = 1.0,
              thinking: bool = False) -> list[dict]:
    """Left-padded batched greedy decoding (as qwen_runtime.generate_batch), optionally with the decay scaling."""
    texts = [qr.prompt_text(rt, 0 if im is None else 1, t, thinking) for im, t in items]
    images = [im for im, _ in items if im is not None]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images or None, padding=True, return_tensors='pt').to(rt.device)
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist() if images else []):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image.')
    mask = batch['input_ids'] == rt.model.config.image_token_id
    with decay_scale(rt, s, mask):
        out = rt.model.generate(**batch, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                                top_k=None)
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = rt.tokenizer.pad_token_id
    res = []
    for row in out[:, batch['input_ids'].shape[1]:]:
        ids = row.tolist()
        end = next((i + 1 for i, t in enumerate(ids) if t in eos), None)
        new = ids[:end] if end is not None else [t for t in ids if t != pad]
        res.append({'text': rt.tokenizer.decode(new, skip_special_tokens=True), 'ids': new, 'n_tokens': len(new),
                    'finish_reason': 'eos' if end is not None else 'length'})
    return res


TEXT_GRID_Q = ('A square canvas is divided into a grid of {n} rows and {n} columns of equal cells. Using 0-1000 '
               'normalized coordinates (x increases rightward and y increases downward), give the center point of '
               'the cell in row {r} and column {c}, counting rows and columns from 1 at the top-left. Return only '
               '[{{"point_2d": [x, y]}}]. Output only the JSON, with no explanation.')


def gen_jobs(cases: Cases, args) -> list[dict]:
    jobs = []
    for fam in cmn.FAMILIES:
        for case in cases.family(fam, args.n_items):
            base = {'family': fam, 'case': case['id']}
            conds = [('point', 1.0, 'clean'), ('point', 0.0, 'clean'), ('bbox', 1.0, 'clean')]
            if fam in ('grid10', 'grid8', 'thumb_mixed', 'thumb_same', 'scatter'):
                conds.append(('point', 1.0, 'delL'))
            if fam == 'grid10':
                conds += [('rowcol', 1.0, 'clean'), ('rowcol', 0.0, 'clean')]
            for fmt, s, var in conds:
                jobs.append(dict(base, model='base', fmt=fmt, s=s, variant=var, _case=case))
            if fam in ('grid10', 'grid8', 'thumb_same', 'thumb_mixed') or (fam == 'refcoco' and
                                                                          case['id'] in cases_tuned_ids(cases, args)):
                tconds = [('point', 1.0, 'clean'), ('point', 0.0, 'clean')]
                if fam == 'grid10':
                    tconds += [('rowcol', 1.0, 'clean'), ('rowcol', 0.0, 'clean'), ('point', 1.0, 'delL')]
                for fmt, s, var in tconds:
                    jobs.append(dict(base, model=LORA_COND, fmt=fmt, s=s, variant=var, _case=case))
    for n in (8, 10):
        for r in range(1, n + 1):
            for c in range(1, n + 1):
                if (r + c) % 2 == 0:
                    jobs.append({'family': f'text{n}', 'case': f'text{n}_r{r}c{c}', 'model': 'base', 'fmt': 'text',
                                 's': 1.0, 'variant': 'clean', '_case': {'n': n, 'r': r, 'c': c}})
    return jobs


_TUNED_IDS: set | None = None


def cases_tuned_ids(cases: Cases, args) -> set:
    global _TUNED_IDS
    if _TUNED_IDS is None:
        _TUNED_IDS = {c['id'] for c in cases.refcoco(args.n_refcoco)[:64]}
    return _TUNED_IDS


def job_prompt(job: dict) -> str:
    case = job['_case']
    if job['fmt'] == 'text':
        return TEXT_GRID_Q.format(**case)
    if job['fmt'] == 'rowcol':
        return cm.prompt('rowcol', 'direct')
    return eh.prompt_for(cmn.target_text(case), job['fmt'])


def stage_gen(rt, args) -> None:
    path = args.output / 'gen.jsonl'
    key = lambda j: (j['family'], j['case'], j['model'], j['fmt'], j['s'], j['variant'])
    done = {key(r) for r in read_jsonl(path)}
    cases = Cases(args)
    jobs = [j for j in gen_jobs(cases, args) if key(j) not in done]
    print(f'gen: {len(jobs)} jobs to run', flush=True)
    lora = None
    for model_name in ('base', LORA_COND):
        mjobs = [j for j in jobs if j['model'] == model_name]
        if not mjobs:
            continue
        if model_name != 'base':
            lora = tf.load_adapter(rt.layers, Path(args.work) / TFR2 / 'adapters' / f'{LORA_COND}.pt', LORA_COND,
                                   model=rt.model)
        groups: dict = {}
        for j in mjobs:
            groups.setdefault((j['fmt'], j['s'], j['family'].startswith('text')), []).append(j)
        for (fmt, s, is_text), js in groups.items():
            max_new = {'point': 48, 'bbox': 64, 'rowcol': 32, 'text': 40}[fmt]
            for k in range(0, len(js), args.batch):
                chunk = js[k:k + args.batch]
                t0 = time.time()
                items, truths = [], []
                for j in chunk:
                    if is_text:
                        items.append((None, job_prompt(j)))
                        truths.append(None)
                    else:
                        img, truth = cases.render(j['_case'], j['variant'])
                        items.append((img, job_prompt(j)))
                        truths.append(truth)
                res = gen_batch(rt, items, max_new, s=s)
                append_jsonl(path, [dict({k2: v for k2, v in j.items() if k2 != '_case'}, **r, truth=t,
                                         batch_size=len(chunk)) for j, r, t in zip(chunk, res, truths)])
                print(f'gen {model_name} {fmt} s={s} {k + len(chunk)}/{len(js)} {time.time() - t0:.1f}s', flush=True)
        if lora is not None:
            lora.detach()
            lora = None


def stage_spot(rt, args) -> None:
    """Batched vs single-sequence greedy decoding on 20 prompts (10 RefCOCO, 10 grid10; point format)."""
    cases = Cases(args)
    sel = cases.family('refcoco', args.n_refcoco)[:10] + cases.family('grid10', args.n_items)[:10]
    items = []
    for c in sel:
        img, _ = cases.render(c)
        items.append((img, eh.prompt_for(cmn.target_text(c), 'point')))
    batched = []
    for k in range(0, len(items), args.batch):
        batched += gen_batch(rt, items[k:k + args.batch], 48)
    single = [qr.generate(rt, qr.build(rt, [im], t), 48) for im, t in items]
    rows = [{'case': c['id'], 'batched': b['text'], 'single': s['text'],
             'same_text': b['text'] == s['text'],
             'same_answer': cmn.parse_point(b['text']) == cmn.parse_point(s['text'])}
            for c, b, s in zip(sel, batched, single)]
    (args.output / 'spotcheck.json').write_text(json.dumps({'rows': rows, 'batch_size': args.batch,
                                                            'same_answer': sum(r['same_answer'] for r in rows),
                                                            'n': len(rows)}, indent=1))


# ----------------------------------------------------------------------------- attention control

class AttnCtl:
    """Image masking and attention recording inside the SDPA call of the decoder full-attention layers.

    mask:   layer -> bool tensor (R,) of rows whose image keys are hidden in this forward call (None = no mask).
    record: if True, store the image attention share (R, 16, n_query) of every full-attention layer, and for
            MAP_LAYERS the attention of the last query over the image tokens (R, 16, n_img).
    """

    def __init__(self, img: tuple[int, int]):
        self.img = img
        self.mask: dict[int, torch.Tensor] | None = None
        self.record = False
        self.share: dict[int, torch.Tensor] = {}
        self.maps: dict[int, torch.Tensor] = {}


_CTL: AttnCtl | None = None
_ORIG = mq.ALL_ATTENTION_FUNCTIONS['sdpa']


def _attn(module, query, key, value, attention_mask, **kwargs):
    ctl = _CTL
    if ctl is None or not isinstance(module, mq.Qwen3_5Attention):
        return _ORIG(module, query, key, value, attention_mask, **kwargs)
    layer = module.layer_idx + 1
    rows_mask = None if ctl.mask is None else ctl.mask.get(layer)
    if rows_mask is None and not ctl.record:
        return _ORIG(module, query, key, value, attention_mask, **kwargs)
    B, _, nq, _ = query.shape
    T = key.shape[2]
    s0, s1 = ctl.img
    if attention_mask is None:
        add = torch.zeros(B, 1, nq, T, device=query.device, dtype=query.dtype)
        if nq > 1:
            i = torch.arange(nq, device=query.device)[:, None]
            j = torch.arange(T, device=query.device)[None, :]
            add = add.masked_fill(j > (T - nq + i), torch.finfo(query.dtype).min)
    elif attention_mask.dtype == torch.bool:
        add = torch.zeros(attention_mask.shape[:-1] + (T,), device=query.device, dtype=query.dtype)
        add = add.masked_fill(~attention_mask[..., :T], torch.finfo(query.dtype).min).expand(B, 1, nq, T).clone()
    else:
        add = attention_mask[..., :T].to(query.dtype).expand(B, 1, nq, T).clone()
    if rows_mask is not None:
        rm = rows_mask.to(query.device)
        add[rm, :, :, s0:s1] = torch.finfo(query.dtype).min
    if ctl.record:
        groups = query.shape[1] // key.shape[1]
        k = key.repeat_interleave(groups, dim=1)
        logits = torch.matmul(query.float(), k.float().transpose(-1, -2)) * module.scaling + add.float()
        w = logits.softmax(-1)                                                 # (B, H, nq, T)
        ctl.share[layer] = w[..., s0:s1].sum(-1).to(torch.float16).cpu()       # (B, H, nq)
        if layer in MAP_LAYERS:
            ctl.maps[layer] = w[:, :, -1, s0:s1].to(torch.float16).cpu()     # (B, H, n_img)
    return _ORIG(module, query, key, value, add, **kwargs)


@contextmanager
def attn_control(ctl: AttnCtl):
    global _CTL
    mq.ALL_ATTENTION_FUNCTIONS['sdpa'] = _attn
    _CTL = ctl
    try:
        yield ctl
    finally:
        _CTL = None
        mq.ALL_ATTENTION_FUNCTIONS['sdpa'] = _ORIG


class HeadSwap:
    """During one forward call, overwrite the o_proj input slices of chosen heads in chosen rows with the slices of
    a source row (all positions of the call). ops: list of (row, layer, head)."""

    def __init__(self, rt, ops: list[tuple[int, int, int]], src: int):
        self.handles = []
        by_layer: dict[int, list[tuple[int, int]]] = {}
        for r, l, h in ops:
            by_layer.setdefault(l, []).append((r, h))
        for l, rh in by_layer.items():
            def pre(m, args, rh=rh):
                x = args[0].clone()
                for r, h in rh:
                    x[r, :, h * HEAD_DIM:(h + 1) * HEAD_DIM] = x[src, :, h * HEAD_DIM:(h + 1) * HEAD_DIM]
                return (x,) + tuple(args[1:])
            self.handles.append(rt.layers[l - 1].self_attn.o_proj.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


# ----------------------------------------------------------------------------- sequence engine

class Seq:
    """Prompt (one or two images of identical size, same text) plus teacher-forced generated ids."""

    def __init__(self, rt, images: list[Image.Image], prompt: str, gen_ids: list[int], thinking: bool = False):
        texts = [qr.prompt_text(rt, 1, prompt, thinking)] * len(images)
        batch = rt.processor(text=texts, images=images, padding=False, return_tensors='pt')
        for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
            if (h * 16, w * 16) != (image.height, image.width):
                raise ValueError('processor resized an image')
        self.rt = rt
        self.P = int(batch['input_ids'].shape[1])
        gen = torch.tensor([gen_ids] * len(images), dtype=batch['input_ids'].dtype)
        self.ids = torch.cat([batch['input_ids'], gen], 1).to(rt.device)
        self.pix = batch['pixel_values'].to(rt.device)
        self.thw = batch['image_grid_thw'].to(rt.device)
        mm = batch.get('mm_token_type_ids')
        self.mm = None if mm is None else torch.cat([mm, torch.zeros_like(gen)], 1).to(rt.device)
        pos = (self.ids[0] == rt.model.config.image_token_id).nonzero().flatten()
        self.img = (int(pos[0]), int(pos[-1]) + 1)

    @torch.inference_mode()
    def prefill(self, upto: int, rows: list[int]):
        """Cache over positions [0, upto) of the distinct inputs, expanded to `rows`; returns (cache, delta (R,))."""
        kw = {'pixel_values': self.pix, 'image_grid_thw': self.thw}
        if self.mm is not None:
            kw['mm_token_type_ids'] = self.mm[:, :upto]
        self.rt.model.model.rope_deltas = None
        o = self.rt.model(input_ids=self.ids[:, :upto], use_cache=True, logits_to_keep=1, **kw)
        cache = o.past_key_values
        delta = self.rt.model.model.rope_deltas.view(-1).clone()
        idx = torch.tensor(rows, device=self.ids.device)
        cache.reorder_cache(idx)
        return cache, delta[idx]

    @torch.inference_mode()
    def forward(self, cache, delta: torch.Tensor, rows: list[int], start: int, end: int) -> torch.Tensor:
        """Feed positions [start, end] for every row (row r uses input row rows[r]); returns last-position logits."""
        R = len(rows)
        ids = self.ids[torch.tensor(rows, device=self.ids.device), start:end + 1]
        ar = torch.arange(start, end + 1, device=ids.device)
        pos = (ar[None, None, :] + delta.view(1, R, 1)).expand(3, R, -1).contiguous()
        return self.rt.model(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True).logits[:, -1].float()


def digit_lp(rt, logits: torch.Tensor) -> np.ndarray:
    ids = torch.tensor([qr.token_id(rt, str(d)) for d in range(10)], device=logits.device)
    return logits.log_softmax(-1)[:, ids].cpu().numpy()


# conditions evaluated with the studied step alone (window = [tau, tau])
STEP_CONDS = ('A', 'B', 'img', 'img@16', 'img@20', 'img@24', 'img@28', 'conv1', 'conv1@25-32', 'conv1@1-24', 'conv',
              'rec', 'delta', 'ans_kv', 'txt_kv', 'prev_all', 'img+prev_all', 'img+delta', 'H11', 'L20', 'L24')
# conditions evaluated over the window [first digit - 3, tau]
WIN_CONDS = ('A', 'B', 'img', 'img@20', 'img@24', 'img@16+20+24', 'H11', 'L16', 'L20', 'L24', 'L28', 'edgeL', 'edgeR',
             'edgeLR', 'allheads')
TAIL_OFFSETS = (-6, -4, -3, -2, -1, 0)


def _heads(spec: str) -> list[tuple[int, int]]:
    if spec == 'H11':
        return list(H11)
    if spec in ('L16', 'L20', 'L24', 'L28'):
        return [(int(spec[1:]), h) for h in range(16)]
    if spec == 'edgeL':
        return list(EDGE_LEFT)
    if spec == 'edgeR':
        return list(EDGE_RIGHT)
    if spec == 'edgeLR':
        return list(EDGE_LEFT + EDGE_RIGHT)
    if spec == 'allheads':
        return [(l, h) for l in FULL_LAYERS for h in range(16)]
    raise ValueError(spec)


@torch.inference_mode()
def run_conditions(seq: Seq, tau: int, w0: int, conds: tuple[str, ...]) -> dict:
    """Digit log-probabilities at step tau (fed position) for each condition; positions [w0, tau] are recomputed."""
    rt = seq.rt
    rows = [1 if c == 'B' else 0 for c in conds]
    cache, delta = seq.prefill(w0, rows)
    full, lin = d2.cache_layers(rt, cache)
    b = conds.index('B')
    s0, s1 = seq.img
    img = list(range(s0, s1))
    ans = list(range(seq.P - 1, w0))           # earlier answer positions (incl. the last prompt token)
    txt = list(range(s1, seq.P - 1))
    ops = []
    for r, c in enumerate(conds):
        for part in c.split('+') if not c.startswith('img@16+') else [c]:
            if part in ('A', 'B'):
                continue
            if part == 'img':
                d2.copy_kv(full, r, b, img, FULL_LAYERS)
            elif part == 'img@16+20+24':
                d2.copy_kv(full, r, b, img, (16, 20, 24))
            elif part.startswith('img@'):
                d2.copy_kv(full, r, b, img, (int(part[4:]),))
            elif part == 'conv1':
                d2.copy_delta(lin, r, b, recurrent=False, last_only=True)
            elif part.startswith('conv1@'):
                lo, hi = (int(v) for v in part[6:].split('-'))
                d2.copy_delta(lin, r, b, recurrent=False, last_only=True, layers=range(lo, hi + 1))
            elif part == 'conv':
                d2.copy_delta(lin, r, b, recurrent=False)
            elif part == 'rec':
                d2.copy_delta(lin, r, b, conv=False)
            elif part == 'delta':
                d2.copy_delta(lin, r, b)
            elif part == 'ans_kv':
                d2.copy_kv(full, r, b, ans, FULL_LAYERS)
            elif part == 'txt_kv':
                d2.copy_kv(full, r, b, txt, FULL_LAYERS)
            elif part == 'prev_all':
                d2.copy_kv(full, r, b, ans, FULL_LAYERS)
                d2.copy_delta(lin, r, b)
            else:
                ops += [(r, l, h) for l, h in _heads(part)]
    with HeadSwap(rt, ops, b):
        logits = seq.forward(cache, delta, rows, w0, tau)
    del cache
    return {'conds': list(conds), 'lp': digit_lp(rt, logits).round(4).tolist()}


@torch.inference_mode()
def run_tail(seq: Seq, tau: int, first: int) -> dict:
    """B's image KV visible to positions >= first + off (off in TAIL_OFFSETS, clipped to <= tau)."""
    out = []
    s0, s1 = seq.img
    starts = sorted({min(first + o, tau) for o in TAIL_OFFSETS})
    for st in starts:
        cache, delta = seq.prefill(st, [0, 1, 0])
        full, _ = d2.cache_layers(seq.rt, cache)
        d2.copy_kv(full, 2, 1, list(range(s0, s1)), FULL_LAYERS)
        out.append(digit_lp(seq.rt, seq.forward(cache, delta, [0, 1, 0], st, tau)).round(4).tolist())
        del cache
    return {'starts': [s - first for s in starts], 'lp': out}


# ----------------------------------------------------------------------------- flow pairs

def disc_pairs(n: int) -> list[dict]:
    rng = np.random.default_rng(1105)
    out = []
    for i in range(n):
        axis = i % 2
        c = rng.uniform(150, 420, 2)
        cb = c.copy()
        cb[axis] += 45                                     # ~70 units on the 640 canvas
        out.append({'id': f'disc_{i:03d}', 'family': 'disc', 'axis': axis, 'cA': c.tolist(), 'cB': cb.tolist(),
                    'target': 'the red circle'})
    return out


def grid_pairs(n: int) -> list[dict]:
    cases = cmn.grid_cases('grid10', n)
    out = []
    for i, c in enumerate(cases):
        axis = i % 2
        r, col = c['target']
        tb = [r + (axis == 1), col + (axis == 0)]
        out.append({'id': f"g10_{c['id']}", 'family': 'grid10', 'axis': axis, 'tA': [r, col], 'tB': tb,
                    'template': c['template'], 'target': cmn.SYNTH_TARGET})
    return out


def refcoco_pairs(cases: Cases, n: int, fmt: str, d_px: int = 28) -> list[dict]:
    out = []
    for c in cases.refcoco(1600):
        if len(out) >= n:
            break
        axis = {'point': len(out) % 2, 'bbox': 0, 'cot': 2}[fmt]
        win = cmn.crop_pair(tuple(c['truth']['canvas']), c['truth']['box'], axis, d_px)
        if win is None:
            continue
        out.append({'id': f"rc_{fmt}_{c['id']}", 'family': f'refcoco_{fmt}', 'axis': axis, 'case': c['id'],
                    'image': c['image'], 'box': c['truth']['box'], 'winA': win[0], 'winB': win[1],
                    'target': c['target']})
    return out


def pair_images(pr: dict, args) -> tuple[Image.Image, Image.Image, dict, dict]:
    fam = pr['family']
    if fam == 'disc':
        a, ta = cr.disc_image(tuple(pr['cA']), 64, 640)
        b, tb = cr.disc_image(tuple(pr['cB']), 64, 640)
        return a, b, cmn.truth_from_box(ta['box'], (640, 640)), cmn.truth_from_box(tb['box'], (640, 640))
    if fam == 'grid10':
        a, ta = cmn.synth_grid(10, 64, tuple(pr['tA']), pr['template'])
        b, tb = cmn.synth_grid(10, 64, tuple(pr['tB']), pr['template'])
        return a, b, ta, tb
    img = Image.open(Path(args.work) / R1_IMAGES / pr['image']).convert('RGB')
    a, b = img.crop(tuple(pr['winA'])), img.crop(tuple(pr['winB']))
    return a, b, cmn.crop_truth(pr['box'], pr['winA']), cmn.crop_truth(pr['box'], pr['winB'])


def flow_pairs(args) -> list[dict]:
    cases = Cases(args)
    return (disc_pairs(args.n_pairs) + grid_pairs(args.n_pairs) + refcoco_pairs(cases, args.n_pairs, 'point')
            + refcoco_pairs(cases, args.n_pairs // 2, 'bbox') + refcoco_pairs(cases, args.n_pairs // 2, 'cot'))


def stage_think(rt, args) -> None:
    """CoT generations on window A of the CoT pairs (batched by window size)."""
    path = args.output / 'think.jsonl'
    done = {r['id'] for r in read_jsonl(path)}
    prs = [p for p in flow_pairs(args) if p['family'] == 'refcoco_cot' and p['id'] not in done]
    groups: dict = {}
    for p in prs:
        a, _, _, _ = pair_images(p, args)
        groups.setdefault(a.size, []).append(p)
    for size, ps in groups.items():
        for k in range(0, len(ps), args.batch):
            chunk = ps[k:k + args.batch]
            items = [(pair_images(p, args)[0], cmn.THINK_PROMPT.format(target=p['target'])) for p in chunk]
            t0 = time.time()
            res = gen_batch(rt, items, 768)
            append_jsonl(path, [dict(r, id=p['id'], batch_size=len(chunk)) for p, r in zip(chunk, res)])
            print(f'think {size} {k + len(chunk)}/{len(ps)} {time.time() - t0:.0f}s', flush=True)


def studied_values(rt, pr: dict, gen_ids: list[int]) -> list[dict]:
    toks = [rt.tokenizer.decode([t]) for t in gen_ids]
    if pr['family'] == 'refcoco_cot':
        runs = [r for r in cmn.digit_runs(toks) if r['len'] == 3]
        fin = cmn.answer_values(toks, 'point')
        vals = []
        if runs and (not fin or runs[0]['start'] < fin[0]['start']):
            vals.append(dict(runs[0], role='cot_first'))
        if fin:
            vals.append(dict(fin[0], role='final_x'))
        return vals
    fmt = 'bbox' if pr['family'] == 'refcoco_bbox' else 'point'
    vals = cmn.answer_values(toks, fmt)
    if fmt == 'bbox':
        vals = [v for v in vals if v['role'] in ('x_min', 'x_max')]
    return [v for v in vals if v['len'] == 3]


def stage_flow(rt, args) -> None:
    path = args.output / 'flow.jsonl'
    done = {r['id'] for r in read_jsonl(path)}
    think = {r['id']: r for r in read_jsonl(args.output / 'think.jsonl')}
    t0 = time.time()
    prs = flow_pairs(args)
    for j, pr in enumerate(prs):
        if pr['id'] in done:
            continue
        a, b, ta, tb = pair_images(pr, args)
        if pr['family'] == 'refcoco_cot':
            if pr['id'] not in think:
                continue
            prompt, gen_ids = cmn.THINK_PROMPT.format(target=pr['target']), think[pr['id']]['ids']
        else:
            fmt = 'bbox' if pr['family'] == 'refcoco_bbox' else 'point'
            prompt = eh.prompt_for(pr['target'], fmt)
            gen_ids = gen_batch(rt, [(a, prompt)], 64)[0]['ids']
        seq = Seq(rt, [a, b], prompt, gen_ids)
        vals = studied_values(rt, pr, gen_ids)
        recs = []
        for v in vals:
            first = seq.P - 1 + v['start']                      # fed position that predicts the hundreds digit
            for i in range(3):
                tau = first + i
                rec = {'role': v['role'], 'digit': i, 'value': v['value']}
                rec['step'] = run_conditions(seq, tau, tau, STEP_CONDS)
                rec['win'] = run_conditions(seq, tau, first - 3, WIN_CONDS)
                rec['tail'] = run_tail(seq, tau, first)
                recs.append(rec)
        append_jsonl(path, [{**{k: v for k, v in pr.items()}, 'truthA': ta, 'truthB': tb,
                             'answerA': rt.tokenizer.decode(gen_ids, skip_special_tokens=True)[-400:],
                             'values': recs}])
        print(f'flow {j + 1}/{len(prs)} {pr["family"]} {len(recs)} steps {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- timeline

TL_ROWS = ('clean', 'm_all', 'm_L20', 'm_L16-24', 'm_cum')


@torch.inference_mode()
def run_timeline(rt, image: Image.Image, prompt: str, gen_ids: list[int], value_steps: list[int]) -> dict:
    """Step through the teacher-forced answer; per step: image attention share (row 'clean'), and log p of the
    teacher-forced token + KL(clean || row) for single-step image hiding and cumulative hiding."""
    seq = Seq(rt, [image], prompt, gen_ids)
    R = len(TL_ROWS)
    cache, delta = seq.prefill(seq.P - 1, [0] * R)
    full, lin = d2.cache_layers(rt, cache)
    ctl = AttnCtl(seq.img)
    steps = []
    n = len(gen_ids)
    with attn_control(ctl):
        for g in range(n):
            tau = seq.P - 1 + g
            ctl.record = True
            m = torch.zeros(R, dtype=torch.bool)
            ctl.mask = {l: m.clone() for l in FULL_LAYERS}
            for l in FULL_LAYERS:
                ctl.mask[l][TL_ROWS.index('m_all')] = True
                ctl.mask[l][TL_ROWS.index('m_cum')] = True
                if l == 20:
                    ctl.mask[l][TL_ROWS.index('m_L20')] = True
                if l in (16, 20, 24):
                    ctl.mask[l][TL_ROWS.index('m_L16-24')] = True
            logits = seq.forward(cache, delta, [0] * R, tau, tau)
            lp = logits.log_softmax(-1)
            nxt = gen_ids[g]                          # step g feeds position P-1+g and predicts generated token g
            fed = int(seq.ids[0, tau])
            rec = {'g': g, 'fed': rt.tokenizer.decode([fed]), 'next': rt.tokenizer.decode([nxt]),
                   'share': np.stack([ctl.share[l][0, :, -1].float().numpy() for l in FULL_LAYERS]).round(4).tolist()}
            rec['logp'] = lp[:, nxt].cpu().numpy().round(4).tolist()
            p0 = lp[0].exp()
            rec['kl'] = ((p0[None] * (lp[0][None] - lp)).sum(-1)).cpu().numpy().round(5).tolist()
            rec['argmax'] = lp.argmax(-1).cpu().tolist()
            if g in value_steps:
                rec['maps'] = {l: ctl.maps[l][0].float().numpy().round(4).tolist() for l in MAP_LAYERS}
            steps.append(rec)
            # single-step rows go back to the clean state; the cumulative row keeps its own history
            for r in range(1, R):
                if TL_ROWS[r] == 'm_cum':
                    continue
                d2.copy_kv(full, r, 0, [tau], FULL_LAYERS)
                d2.copy_delta(lin, r, 0)
    del cache
    return {'P': seq.P, 'img': list(seq.img), 'steps': steps}


def timeline_jobs(args) -> list[dict]:
    cases = Cases(args)
    jobs = []
    for fam in cmn.FAMILIES:
        for c in cases.family(fam, args.n_items)[:args.n_timeline]:
            jobs.append({'id': f'tl_{c["id"]}_point', 'family': fam, 'fmt': 'point', '_case': c})
            if fam == 'refcoco':
                jobs.append({'id': f'tl_{c["id"]}_bbox', 'family': fam, 'fmt': 'bbox', '_case': c})
    for p in flow_pairs(args):
        if p['family'] == 'refcoco_cot':
            jobs.append({'id': f'tl_{p["id"]}', 'family': 'refcoco', 'fmt': 'cot', '_pair': p})
    return jobs


def stage_timeline(rt, args) -> None:
    path = args.output / 'timeline.jsonl'
    done = {r['id'] for r in read_jsonl(path)}
    think = {r['id']: r for r in read_jsonl(args.output / 'think.jsonl')}
    cases = Cases(args)
    t0 = time.time()
    jobs = timeline_jobs(args)
    for j, job in enumerate(jobs):
        if job['id'] in done:
            continue
        if job['fmt'] == 'cot':
            p = job['_pair']
            if p['id'] not in think:
                continue
            img, _, truth, _ = pair_images(p, args)
            prompt, ids = cmn.THINK_PROMPT.format(target=p['target']), think[p['id']]['ids']
            toks = [rt.tokenizer.decode([t]) for t in ids]
            vals = [dict(r, role='num') for r in cmn.digit_runs(toks) if r['len'] == 3]
        else:
            img, truth = cases.render(job['_case'])
            prompt = eh.prompt_for(cmn.target_text(job['_case']), job['fmt'])
            ids = gen_batch(rt, [(img, prompt)], 64)[0]['ids']
            toks = [rt.tokenizer.decode([t]) for t in ids]
            vals = cmn.answer_values(toks, job['fmt'])
        # loop step g predicts generated token g, so a value's first-digit step is g = start
        vsteps = [v['start'] for v in vals]
        res = run_timeline(rt, img, prompt, ids, vsteps)
        append_jsonl(path, [{'id': job['id'], 'family': job['family'], 'fmt': job['fmt'], 'truth': truth,
                             'values': vals, 'text': rt.tokenizer.decode(ids, skip_special_tokens=True)[-600:],
                             **res}])
        print(f'timeline {j + 1}/{len(jobs)} {job["id"]} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--model', default='Qwen3.5-9B')
    ap.add_argument('--n-items', type=int, default=64)
    ap.add_argument('--n-refcoco', type=int, default=160)
    ap.add_argument('--n-pairs', type=int, default=48)
    ap.add_argument('--n-timeline', type=int, default=24)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    meta = {**qr.describe(rt), 'stages': args.stages, 'code': code_digest(), 'n_items': args.n_items,
            'n_refcoco': args.n_refcoco, 'n_pairs': args.n_pairs, 'n_timeline': args.n_timeline,
            'batch_size': args.batch, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    (args.output / f"run_{time.strftime('%m%d_%H%M')}.json").write_text(json.dumps(meta, indent=1))
    stages = {'spot': stage_spot, 'gen': stage_gen, 'think': stage_think, 'flow': stage_flow,
              'timeline': stage_timeline}
    for s in args.stages.split(','):
        print(f'== stage {s}', flush=True)
        stages[s](rt, args)


if __name__ == '__main__':
    main()
