"""Experiment 3.11 round 5: how two edge reads become a centre (the factor 0.5), and why L20 H11 is the dominant
reader on synthetic images. Qwen3.5-9B and Qwen3-VL-8B (mechanism only).

Results: docs/analysis/position/results_2026-11-11_edge_heads_r3.md. Builds on round 4 (edge_heads_r2.py), whose
item sets (str, rect, nat) and round-3 sources are reused; images are rendered deterministically from the specs.

GPU stages (one process, each resumable by record id):
  avg     one-edge pairs (str, rect; one side expanded by 96 px) in point and box format. (1) Greedy decoding of
          run A, run B and run A with head groups transplanted from run B at every answer step (batch rows; the
          head output of row B at the same decoding step is copied, i.e. activation patching during decoding):
          the slope of the moved value per group in pixels. (2) Teacher-forced run A / run B at the query of the
          moved value's first digit: every full-attention head output (o_proj input slice) of both runs, first-digit
          logit recovery r of the groups, and (Qwen3.5) attribution patching of every MLP neuron of layers 24-30
          (activation difference B - A times the gradient of the first-digit logit difference in run A, summed over
          the answer positions up to the query), verified by patching the top-k neurons from B.
  vlscan  (Qwen3-VL) round-4 nat translation pairs, point and box: image-attention region masses of every head at
          every first-digit query, then single-head patching of the heads with the largest image share.
  mix     where pairs on interpolations between synthetic and natural images (shapes / textured shapes / cut-outs
          on white, grey, noise or photo backgrounds; 10x10 grids). Attention of every head, head-group patching,
          image key/value splicing by region (target, ring, row right / left, column, rest, all) in every
          full-attention layer, single-head scan, and (Qwen3.5) attention-pattern transplants between L20 H11 and
          the edge heads.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import causal_vs_linear as cv
from analysis_tools.position import causal_vs_linear_heads as cvh
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_mechanism_natural as cmn
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import edge_heads as eh
from analysis_tools.position import edge_heads_r2 as e2

sr = cf.sr
read_jsonl, append_jsonl, done_ids = sr.read_jsonl, sr.append_jsonl, sr.done_ids

TP = cr.TOKEN_PIXELS
MLP_LAYERS = tuple(range(24, 31))          # Qwen3.5 readout MLPs (round 4: MLP 24-30 carry the edge sum to digits)
TOPK = (8, 32, 128, 512, 2048, 8192)       # neuron counts for the top-k verification
RAND_K = 2048
SIDE_KEY = {('point', 'right'): 'x', ('point', 'left'): 'x', ('point', 'top'): 'y', ('point', 'bottom'): 'y',
            ('bbox', 'right'): 'x_max', ('bbox', 'left'): 'x_min', ('bbox', 'top'): 'y_min', ('bbox', 'bottom'): 'y_max'}
# Group that reads the moved edge at a point answer, and the group that reads the opposite edge.
SIDE_GROUPS = {'right': ('maxx', 'minx'), 'left': ('minx', 'maxx'), 'bottom': ('maxy', 'miny'), 'top': ('miny', 'maxy')}
REGIONS = ('target', 'ring', 'row_right', 'row_left', 'column', 'rest', 'all')
MIX_KINDS = ('shape_white', 'shape_grey', 'shape_noise', 'shape_photo', 'texshape_white',
             'cutout_white', 'cutout_grey', 'cutout_photo')
GRID_KIND = 'grid10'
MIX_SIZE = 96
H11 = (20, 11)
GRID_TARGET = cmn.SYNTH_TARGET
cfn = e2.cfn


# ----------------------------------------------------------------------------- model-family helpers

def full_layers(rt) -> tuple[int, ...]:
    return cv.full_layers(rt)


def n_heads(rt) -> int:
    return cv.n_heads(rt)


def parse_heads(spec: str) -> list[tuple[int, int]]:
    """'layer.head,...' (1-based layers) without model-specific range checks; Ctx validates against the model."""
    out = []
    for t in filter(None, spec.split(',')):
        l, h = t.split('.')
        out.append((int(l), int(h)))
    return out


def hname(l: int, h: int) -> str:
    return f'L{l}H{h}'


class HeadCopy:
    """Activation patching between batch rows during prefill (positions >= start) and every decoding step.

    ops: (row, layer, head, src_row). The o_proj input slice of `head` in `row` is overwritten with the same slice
    of `src_row` (the unedited value; source rows must not be edited). Works for Qwen3.5 and Qwen3-VL.
    """

    def __init__(self, rt, ops: list[tuple[int, int, int, int]], start: int):
        self.handles = []
        by_layer: dict[int, list] = {}
        for op in ops:
            by_layer.setdefault(op[1], []).append(op)
        for layer, lops in by_layer.items():
            module, hd = cvh.head_projection(rt, layer)

            def pre(module, args, lops=lops, hd=hd):
                x = args[0]
                lo = start if x.shape[1] > 1 else 0
                y = x.clone()
                for row, _, h, src in lops:
                    sl = slice(h * hd, (h + 1) * hd)
                    y[row, lo:, sl] = x[src, lo:, sl]
                return (y,)
            self.handles.append(module.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


class TailCapture:
    """Batch row 0 of one forward: o_proj inputs at positions >= start (full-attention layers), MLP down_proj
    inputs (neuron activations) at positions >= start (mlp_layers), and k_proj / v_proj outputs on the image span
    (kv_layers)."""

    def __init__(self, rt, start: int, o_layers=(), mlp_layers=(), kv_layers=(), span=(0, 0)):
        self.o, self.act, self.k, self.v = {}, {}, {}, {}
        self.handles = []
        s, e = span
        for l in o_layers:
            mod, _ = cvh.head_projection(rt, l)
            self.handles.append(mod.register_forward_pre_hook(
                lambda m, a, l=l: self.o.__setitem__(l, a[0][0, start:].detach().clone())))
        for l in mlp_layers:
            mod = rt.layers[l - 1].mlp.down_proj
            self.handles.append(mod.register_forward_pre_hook(
                lambda m, a, l=l: self.act.__setitem__(l, a[0][0, start:].detach().clone())))
        for l in kv_layers:
            att = rt.layers[l - 1].self_attn
            self.handles.append(att.k_proj.register_forward_hook(
                lambda m, a, o, l=l: self.k.__setitem__(l, o[0, s:e].detach().clone())))
            self.handles.append(att.v_proj.register_forward_hook(
                lambda m, a, o, l=l: self.v.__setitem__(l, o[0, s:e].detach().clone())))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


class SetEdit:
    """Row-wise replacements inside one teacher-forced batched forward.

    ops (kind, row, layer, a, b):
      ('head', row, layer, head, value (T, hd))        o_proj input slice of one head at positions >= start
      ('mlpn', row, layer, idx (n,), value (T, n))      down_proj input (neuron activations) at positions >= start
      ('kv', row, layer, pos (m,), (k (m, dk), v (m, dv)))  k_proj / v_proj outputs at absolute positions pos
    """

    def __init__(self, rt, ops: list[tuple], start: int):
        self.handles = []
        groups: dict[tuple[str, int], list] = {}
        for op in ops:
            groups.setdefault((op[0], op[2]), []).append(op)
        for (kind, layer), lops in groups.items():
            if kind == 'head':
                module, hd = cvh.head_projection(rt, layer)

                def pre(m, args, lops=lops, hd=hd):
                    y = args[0].clone()
                    for _, row, _, h, val in lops:
                        y[row, start:, h * hd:(h + 1) * hd] = val.to(y.dtype)
                    return (y,)
                self.handles.append(module.register_forward_pre_hook(pre))
            elif kind == 'mlpn':
                module = rt.layers[layer - 1].mlp.down_proj

                def pre(m, args, lops=lops):
                    y = args[0].clone()
                    for _, row, _, idx, val in lops:
                        y[row, start:, idx] = val.to(y.dtype)
                    return (y,)
                self.handles.append(module.register_forward_pre_hook(pre))
            elif kind == 'kv':
                att = rt.layers[layer - 1].self_attn
                for which, mod in ((0, att.k_proj), (1, att.v_proj)):
                    def hook(m, args, out, lops=lops, which=which):
                        y = out.clone()
                        for _, row, _, pos, kv in lops:
                            y[row, pos] = kv[which].to(y.dtype)
                        return y
                    self.handles.append(mod.register_forward_hook(hook))
            else:
                raise ValueError(kind)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def forward_logits(rt, x, pos, logit_pos: list[int], ds_rows=None, span=(0, 0)) -> torch.Tensor:
    with cv.deepstack_rows(rt, ds_rows, span):
        return rt.model(inputs_embeds=x, position_ids=pos, use_cache=False,
                        logits_to_keep=torch.tensor(logit_pos, device=x.device)).logits.float()


def batched_rows(rt, x, pos, rows: list[list[tuple]], start: int, logit_pos: list[int], chunk: int,
                 ds=None, span=(0, 0)) -> torch.Tensor:
    """Logits (R, len(logit_pos), V) of run A (x, pos) under each row's SetEdit ops (row index inside the op is
    rewritten to the chunk-local row)."""
    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        ops = [(op[0], j, *op[2:]) for j, r in enumerate(part) for op in r]
        R = len(part)
        with SetEdit(rt, ops, start):
            out.append(forward_logits(rt, x.expand(R, -1, -1).contiguous(), pos.expand(-1, R, -1).contiguous(),
                                      logit_pos, None if ds is None else [ds], span))
    return torch.cat(out, 0)


def first_digit_ld(lg_rows: torch.Tensor, lg_b: torch.Tensor, ans: list[int], vals: list[dict], digits: list[int]
                   ) -> list[dict]:
    return e2._ld(lg_rows, lg_b, ans, vals, digits)


def head_internals(rt, x, pos, layers, start: int, ds=None, span=(0, 0)) -> dict:
    """Qwen3.5 only. For batch row 0 of one forward and each listed full-attention layer: attention probabilities
    of every head for the queries at positions >= start over all keys (H, T, L), value vectors of every key
    (KV, L, hd), output-gate pre-activations (T, H, hd), and the model's own o_proj input (T, H, hd) for checking."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    store, handles = {}, []
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn

        def pre(module, args, kwargs, layer=layer):
            store[layer] = {'h': (kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0])[:1],
                            'pe': kwargs['position_embeddings']}

        def opre(module, args, layer=layer):
            store[layer]['o'] = args[0][0, start:].detach()
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(mod.o_proj.register_forward_pre_hook(opre))
    try:
        rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    out = {}
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        h = store[layer]['h']
        cos, sin = store[layer]['pe']
        cos, sin = cos[:1], sin[:1]
        L, hd = h.shape[1], mod.head_dim
        q, gate = torch.chunk(mod.q_proj(h[:, start:]).view(1, L - start, -1, hd * 2), 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(1, L, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(1, L, -1, hd).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, start:], sin[:, start:])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        kr = repeat_kv(k, mod.num_key_value_groups)
        sc = (q.float() @ kr.float().transpose(2, 3)) * mod.scaling
        qpos = torch.arange(start, L, device=h.device)
        sc = sc.masked_fill(torch.arange(L, device=h.device)[None, :] > qpos[:, None], float('-inf'))
        a = sc.softmax(-1)[0]                                                   # (H, T, L)
        out[layer] = {'a': a, 'v': v[0].float(), 'gate': gate[0].float(), 'o': store[layer]['o'].float().view(L - start, -1, hd),
                      'groups': mod.num_key_value_groups}
    return out


def pattern_output(intern: dict, layer: int, head: int, pattern: torch.Tensor) -> torch.Tensor:
    """Output (T, hd) of head (layer, head) when its attention probabilities are replaced by `pattern` (T, L):
    sigmoid(gate) * (pattern @ v), with the head's own values and output gate (Qwen3.5 gated attention)."""
    d = intern[layer]
    v = d['v'][head // d['groups']]                                              # (L, hd)
    return torch.sigmoid(d['gate'][:, head]) * (pattern.float() @ v)


# ----------------------------------------------------------------------------- mix items (CPU)

def mix_background(spec: dict, r3: dict) -> Image.Image:
    bg = spec['bg']
    C = e2.CANVAS
    if bg == 'white':
        return Image.new('RGB', (C, C), (255, 255, 255))
    if bg == 'grey':
        return Image.new('RGB', (C, C), (150, 150, 150))
    if bg == 'noise':
        rng = np.random.default_rng(spec['seed'])
        n = rng.normal(150, 45, (C, C, 3)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(n).filter(ImageFilter.GaussianBlur(1.5))
    if bg == 'photo':
        return e2.background({'bg': 'photo', 'bg_src': spec['bg_src']}, r3)
    raise ValueError(bg)


def draw_textured(img: Image.Image, obj: dict, box, seed: int) -> None:
    """Coloured shape whose fill is modulated by a smooth random texture (factor 0.45-1.0)."""
    C = e2.CANVAS
    mask = Image.new('L', (C, C), 0)
    cfn.draw_shape(ImageDraw.Draw(mask), obj['shape'], tuple(box), 255)
    rng = np.random.default_rng(seed)
    tex = Image.fromarray((rng.random((C, C)) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.5))
    t = np.asarray(tex, float) / 255
    t = 0.45 + 0.55 * (t - t.min()) / max(1e-6, t.max() - t.min())
    col = np.array(e2.SHAPE_COLORS[obj['color']], float)
    fill = (t[..., None] * col[None, None]).clip(0, 255).astype(np.uint8)
    img.paste(Image.fromarray(fill), (0, 0), mask)


def render_mix(spec: dict, r3: dict, moved: bool) -> tuple[Image.Image, list[float]]:
    if spec['kind'] == GRID_KIND:
        tgt = spec['target_b'] if moved else spec['target']
        img, truth = cmn.synth_grid(10, 64, tuple(tgt), spec['template'])
        return img, list(truth['box'])
    img = mix_background(spec, r3)
    for i, o in enumerate(spec['objects']):
        box = e2.shift_box(o['box'], *spec['move']) if (moved and i == 0) else o['box']
        if o['kind'] == 'shape' and spec.get('textured'):
            draw_textured(img, o, box, spec['seed'] + i)
        else:
            e2.draw_object(img, o, box, r3)
    box = e2.shift_box(spec['box'], *spec['move']) if moved else spec['box']
    return img, box


def mix_cases(rng, r3: dict, n: int, n_grid: int) -> list[dict]:
    cut_src = [c for c in r3.values() if c['category'] in e2.CUTOUT_CATEGORIES and c['group'] != 'rc'
               and c.get('mask_path') and c['area_frac'] >= 0.04 and e2._interior(c)]
    bg_src = [c['id'] for c in r3.values() if c['family'] in e2.BG_FAMILIES and c['group'] in ('paco', 'stuff')]
    cases, k = [], 0
    for kind in MIX_KINDS:
        obj, bg = kind.split('_')
        objtype = 'cutout' if obj == 'cutout' else 'shape'
        for nd in (0, 2):
            made = 0
            while made < n:
                spec = e2.comp_spec(rng, objtype, 'plain', MIX_SIZE, nd, cut_src, bg_src)
                if spec is None:
                    continue
                spec.update({'id': f'mix_{k:04d}', 'set': 'mix', 'kind': kind, 'bg': bg, 'textured': obj == 'texshape',
                             'seed': int(rng.integers(1 << 30)), 'bg_src': str(rng.choice(bg_src)) if bg == 'photo' else None})
                cases.append(spec)
                k += 1
                made += 1
    for i in range(n_grid):
        r, c = (int(v) for v in rng.integers(1, 8, 2))
        dr, dc = (int(v) for v in rng.choice([-1, 1], 2))
        cases.append({'id': f'mix_{k:04d}', 'set': 'mix', 'kind': GRID_KIND, 'bg': 'white', 'ndistr': 99,
                      'target': [r, c], 'target_b': [r + dr, c + dc], 'template': cmn.TEMPLATES[i % len(cmn.TEMPLATES)],
                      'expr': GRID_TARGET, 'canvas': [640, 640]})
        k += 1
    return cases


def stage_cases(out: Path, args) -> None:
    rng = np.random.default_rng(args.seed)
    r3 = eh.load_cases(args.round3)
    cases = mix_cases(rng, r3, args.n_mix, args.n_grid)
    for c in cases:
        if c['kind'] == GRID_KIND:
            _, c['box'] = render_mix(c, r3, False)
            _, c['box_b'] = render_mix(c, r3, True)
    (out / 'mix_cases.json').write_text(json.dumps(cases))
    from collections import Counter
    print(Counter(c['kind'] for c in cases))


def token_regions(box_a, box_b, grid: tuple[int, int]) -> dict[str, np.ndarray]:
    """Image-token index sets (row-major in the merged grid) for the splice: target = tokens overlapping either
    box; ring = the 1-token ring around the target; row_right / row_left = remaining tokens of the target's token
    rows to the right / left of the target; column = remaining tokens of the target's token columns; rest."""
    Hn, Wn = grid
    t = np.zeros((Hn, Wn), bool)
    for b in (box_a, box_b):
        c0, r0 = int(b[0] // TP), int(b[1] // TP)
        c1, r1 = int(min(Wn - 1, (b[2] - 1e-6) // TP)), int(min(Hn - 1, (b[3] - 1e-6) // TP))
        t[r0:r1 + 1, c0:c1 + 1] = True
    rows, cols = np.nonzero(t)
    pad = np.pad(t, 1)
    ring = np.zeros_like(t)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            ring |= pad[dr:dr + Hn, dc:dc + Wn]
    ring &= ~t
    used = t | ring
    rr = np.zeros_like(t)
    rr[rows.min():rows.max() + 1] = True
    cc = np.zeros_like(t)
    cc[:, cols.min():cols.max() + 1] = True
    right = rr & ~used & (np.arange(Wn)[None] > cols.max())
    left = rr & ~used & (np.arange(Wn)[None] < cols.min())
    col = cc & ~used & ~rr
    rest = ~(used | right | left | col)
    flat = lambda m: np.flatnonzero(m.ravel())
    return {'target': flat(t), 'ring': flat(ring), 'row_right': flat(right), 'row_left': flat(left),
            'column': flat(col), 'rest': flat(rest), 'all': np.arange(Hn * Wn)}


# ----------------------------------------------------------------------------- context

class Ctx:
    def __init__(self, rt, out: Path, args):
        self.rt, self.out, self.args = rt, out, args
        self.r2 = e2.load_cases(args.round4)
        self.r3 = eh.load_cases(args.round3)
        mp = out / 'mix_cases.json'
        self.mix = {c['id']: c for c in json.loads(mp.read_text())} if mp.exists() else {}
        self.digits = cf.digit_ids(rt)
        self.groups = {k: parse_heads(getattr(args, f'group_{k}')) for k in ('minx', 'maxx', 'miny', 'maxy')}
        self.FL = full_layers(rt)
        self.H = n_heads(rt)
        bad = [h for g in self.groups.values() for h in g if h[0] not in self.FL or not 0 <= h[1] < self.H]
        if bad:
            raise ValueError(f'Heads not in this model: {bad}')
        self.all_heads = [(l, h) for l in self.FL for h in range(self.H)]
        self.q35 = rt.family == 'qwen3_5'
        self.tag = 'q35' if self.q35 else 'vl'

    def prep(self, img, text):
        return cv.prepare(self.rt, img, text)

    def ds(self, prep):
        return prep.meta.get('ds') if not self.q35 else None


def span_of(prep) -> tuple[int, int]:
    return prep.span['start'], prep.span['end']


def value_rows(rt, results: list[dict], fmt: str) -> list[dict | None]:
    out = []
    for r in results:
        v = cf.answer_values(rt, r['ids'], fmt)
        out.append(None if v is None else {d['key']: d['value'] for d in v})
    return out


# ----------------------------------------------------------------------------- stage: avg

def avg_jobs(ctx: Ctx) -> list[tuple[dict, str, str]]:
    jobs = []
    for c in ctx.r2.values():
        if c['set'] in ('str', 'rect'):
            for side, delta in c['scan']:
                for f in ('point', 'bbox'):
                    jobs.append((c, side, f))
    return jobs


def avg_groups(ctx: Ctx, side: str) -> dict[str, list[tuple[int, int]]]:
    moved, other = SIDE_GROUPS[side]
    g = {'grp_moved': ctx.groups[moved], 'grp_other': ctx.groups[other],
         'both': ctx.groups[moved] + ctx.groups[other], 'all': ctx.all_heads}
    if ctx.q35:
        g['H11'] = [H11]
        g['all_but_edges'] = [h for h in ctx.all_heads if h not in set(g['both'])]
    return g


@torch.inference_mode()
def avg_generate(ctx: Ctx, pa, pb, groups: dict) -> list[dict]:
    rt = ctx.rt
    names = ['A', 'B'] + list(groups)
    xa, xb = cr.embeds(rt, pa), cr.embeds(rt, pb)
    X = torch.cat([xa, xb] + [xa] * len(groups), 0)
    P = pa.pos.expand(-1, len(names), -1).contiguous()
    ops = [(j + 2, l, h, 1) for j, n in enumerate(groups) for l, h in groups[n]]
    L = len(pa.ids)
    ds = None if ctx.q35 else [ctx.ds(pa), ctx.ds(pb)] + [ctx.ds(pa)] * len(groups)
    with HeadCopy(rt, ops, L - 1), cv.deepstack_rows(rt, ds, span_of(pa)):
        res = cr.greedy(rt, X, P, ctx.args.max_new_tokens)
    return [{'cond': n, 'text': r['text'], 'ids': r['ids']} for n, r in zip(names, res)]


def stage_avg(ctx: Ctx) -> None:
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / f'avg_{ctx.tag}.jsonl'
    adir = out / f'avg_{ctx.tag}'
    adir.mkdir(exist_ok=True)
    finished = done_ids(path)
    for c, side, f in avg_jobs(ctx)[:ctx.args.limit or None]:
        jid = f"{c['id']}:{side}:{f}"
        if jid in finished:
            continue
        t0 = time.time()
        img_a, box_a = e2.render(c, ctx.r3)
        img_b, box_b = e2.render(c, ctx.r3, (side, 96))
        prompt = eh.prompt_for(c['expr'], f)
        pa, pb = ctx.prep(img_a, prompt), ctx.prep(img_b, prompt)
        if len(pa.ids) != len(pb.ids):
            raise ValueError(f'Prompt lengths differ for {jid}')
        groups = avg_groups(ctx, side)
        gen = avg_generate(ctx, pa, pb, groups)
        for g in gen:
            g['values'] = value_rows(rt, [g], f)[0]
        rec = {'id': jid, 'case': c['id'], 'set': c['set'], 'side': side, 'format': f, 'canvas': c['canvas'],
               'box_a': box_a, 'box_b': box_b, 'gen': gen, 'groups': {k: [hname(*h) for h in v] for k, v in groups.items()
                                                                       if k != 'all' and k != 'all_but_edges'}}
        vals = cf.answer_values(rt, gen[0]['ids'], f)
        if vals is not None:
            tf, arrays = avg_teacher_forced(ctx, pa, pb, gen[0]['ids'], vals, SIDE_KEY[(f, side)], groups)
            rec.update(tf)
            np.savez_compressed(adir / f"{jid.replace(':', '__')}.npz", **arrays)
        else:
            rec['tf_skipped'] = 'run A answer not parseable'
        rec['seconds'] = round(time.time() - t0, 1)
        append_jsonl(path, [rec])
        print(f'avg {jid} {rec["seconds"]}s', flush=True)


def avg_teacher_forced(ctx: Ctx, pa, pb, ans: list[int], vals: list[dict], key: str, groups: dict
                       ) -> tuple[dict, dict]:
    rt, args = ctx.rt, ctx.args
    L = len(pa.ids)
    start = L - 1
    v = next(d for d in vals if d['key'] == key)
    q = L - 1 + v['toks'][0]
    tq = q - start                                   # index of the query inside the captured tail
    xa, pos = cr.teacher_forced(rt, pa, ans[:-1])
    xb, _ = cr.teacher_forced(rt, pb, ans[:-1])
    span = span_of(pa)
    mlp = MLP_LAYERS if ctx.q35 else ()
    dsa, dsb = ctx.ds(pa), ctx.ds(pb)
    with torch.inference_mode():
        with TailCapture(rt, start, ctx.FL, mlp) as ca:
            lg_a = forward_logits(rt, xa, pos, [q], None if dsa is None else [dsa], span)
        with TailCapture(rt, start, ctx.FL, mlp) as cb:
            lg_b = forward_logits(rt, xb, pos, [q], None if dsb is None else [dsb], span)
    hd = ca.o[ctx.FL[0]].shape[-1] // ctx.H
    heads_a = torch.stack([ca.o[l][tq].view(ctx.H, hd) for l in ctx.FL]).float().cpu().numpy()
    heads_b = torch.stack([cb.o[l][tq].view(ctx.H, hd) for l in ctx.FL]).float().cpu().numpy()
    one = [v]
    rows = [[]]
    names = ['none'] + list(groups)
    for n in groups:
        rows.append([('head', 0, l, h, cb.o[l][:, h * hd:(h + 1) * hd]) for l, h in groups[n]])
    rec = {'query': q, 'key': key}
    arrays = {'heads_a': heads_a.astype(np.float16), 'heads_b': heads_b.astype(np.float16)}
    if ctx.q35:
        rows.append([('mlpn', 0, l, torch.arange(ca.act[l].shape[-1], device=xa.device), cb.act[l]) for l in MLP_LAYERS])
        names.append('mlp24_30')
    with torch.inference_mode():
        lg = batched_rows(rt, xa, pos, rows, start, [q], args.cond_batch, dsa, span)
    lds = first_digit_ld(lg, lg_b[0], ans, one, ctx.digits)[0]
    rec.update({'conds': names, 'ld': lds})
    if not ctx.q35:
        return rec, arrays
    # attribution patching over MLP 24-30 neurons at positions start..q
    dA = ans[v['toks'][0]]
    dB = ctx.digits[int(lg_b[0, 0, ctx.digits].argmax())]
    acts = {}
    handles = []
    for l in MLP_LAYERS:
        def pre(m, a, l=l):
            a[0].retain_grad()
            acts[l] = a[0]
        handles.append(rt.layers[l - 1].mlp.down_proj.register_forward_pre_hook(pre))
    try:
        with torch.enable_grad():
            x = xa.detach().clone().requires_grad_(True)
            lgg = rt.model(inputs_embeds=x, position_ids=pos, use_cache=False,
                           logits_to_keep=torch.tensor([q], device=x.device)).logits.float()
            ld = lgg[0, 0, dB] - lgg[0, 0, dA]
            ld.backward()
    finally:
        for h in handles:
            h.remove()
    attr, grad_q, dact_q, pos_tot = [], [], [], []
    for l in MLP_LAYERS:
        g = acts[l].grad[0, start:q + 1].float()                                 # (tq+1, I)
        da = (cb.act[l][:tq + 1] - ca.act[l][:tq + 1]).float()
        a = (g * da)
        attr.append(a.sum(0))
        pos_tot.append(a.sum(1))
        grad_q.append(g[tq])
        dact_q.append(da[tq])
    attr = torch.stack(attr)                                                    # (7, I)
    rt.model.zero_grad(set_to_none=True)
    del acts
    # top-k verification: patch the k neurons with the largest |attribution| (positions >= start) from run B
    I = attr.shape[1]
    order = torch.argsort(attr.abs().flatten(), descending=True)
    rows, names_k = [], []
    gen = torch.Generator().manual_seed(int(q) * 7919 + I)
    sets = [(f'top{k}', order[:k]) for k in TOPK] + [(f'rand{RAND_K}', torch.randperm(attr.numel(), generator=gen)[:RAND_K].to(order.device))]
    for name, flat in sets:
        r = []
        for li, l in enumerate(MLP_LAYERS):
            idx = flat[(flat // I) == li] % I
            if len(idx):
                r.append(('mlpn', 0, l, idx, cb.act[l][:, idx]))
        rows.append(r)
        names_k.append(name)
    with torch.inference_mode():
        lg = batched_rows(rt, xa, pos, rows, start, [q], args.cond_batch)
    ldk = first_digit_ld(lg, lg_b[0], ans, one, ctx.digits)[0]
    rec.update({'topk_conds': names_k, 'topk_ld': ldk, 'atp_ld': round(float(ld.detach()), 4),
                'atp_sum': round(float(attr.sum()), 4), 'atp_pos_total': [[round(float(t), 4) for t in p] for p in torch.stack(pos_tot).cpu()]})
    arrays.update({'attr': attr.cpu().numpy().astype(np.float16), 'grad_q': torch.stack(grad_q).cpu().numpy().astype(np.float16),
                   'dact_q': torch.stack(dact_q).cpu().numpy().astype(np.float16)})
    return rec, arrays


# ----------------------------------------------------------------------------- stage: vlscan (and attention region masses)

def region_masses(att_img: np.ndarray, labels: np.ndarray, nreg: int) -> np.ndarray:
    """att_img (..., Nv) -> (..., nreg) attention mass per region label."""
    out = np.zeros(att_img.shape[:-1] + (nreg,), np.float32)
    for r in range(nreg):
        out[..., r] = att_img[..., labels == r].sum(-1)
    return out


def attention_at(ctx: Ctx, x, pos, queries: list[int], prep) -> np.ndarray:
    """(nFullLayers, H, nq, Nv) attention over the image tokens at the given queries (batch row 0)."""
    s, e = span_of(prep)
    rows = cv.attention_rows(ctx.rt, x, pos, queries, ds=ctx.ds(prep), span=(s, e))
    return np.stack([rows[l][:, :, s:e].numpy() for l in ctx.FL])


def stage_vlscan(ctx: Ctx) -> None:
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / f'scan_{ctx.tag}.jsonl'
    finished = done_ids(path)
    gen_path = out / f'nat_gen_{ctx.tag}.jsonl'
    rdir = out / f'scan_{ctx.tag}_regions'
    rdir.mkdir(exist_ok=True)
    nat = [c for c in ctx.r2.values() if c['set'] == 'nat'][:ctx.args.limit or None]
    # phase 1: greedy answers and attention region masses for every nat item and format
    gdone = done_ids(gen_path)
    for c in nat:
        for f in ('point', 'bbox'):
            gid = f"{c['id']}:{f}"
            if gid in gdone:
                continue
            img, box = e2.render(c, ctx.r3)
            p = ctx.prep(img, eh.prompt_for(c['expr'], f))
            with cv.deepstack_rows(rt, None if ctx.q35 else [ctx.ds(p)], span_of(p)):
                r = cr.greedy(rt, cr.embeds(rt, p), p.pos, args.max_new_tokens)[0]
            vals = cf.answer_values(rt, r['ids'], f)
            rec = {'id': gid, 'case': c['id'], 'format': f, 'text': r['text'], 'ids': r['ids'], 'grid': list(p.grid),
                   'values': vals, 'box': box}
            if vals is not None:
                L = len(p.ids)
                x, pos = cr.teacher_forced(rt, p, r['ids'][:-1])
                qs = [L - 1 + v['toks'][0] for v in vals]
                att = attention_at(ctx, x, pos, qs, p)                       # (nL, H, nq, Nv)
                labels = e2.region_labels(box, p.grid)
                rm = region_masses(att, labels, len(e2.REGIONS))                # (nL, H, nq, nreg)
                rec['image_share'] = att.sum(-1).mean(-1).astype(np.float32).round(4).tolist()   # (nL, H)
                np.savez_compressed(rdir / f"{gid.replace(':', '__')}.npz", regions=rm.astype(np.float16),
                                    share=att.sum(-1).astype(np.float16))
            append_jsonl(gen_path, [rec])
        print(f'vlscan gen {c["id"]}', flush=True)
    gens = {g['id']: g for g in read_jsonl(gen_path)}
    share = np.mean([g['image_share'] for g in gens.values() if 'image_share' in g], 0)   # (nL, H)
    order = np.argsort(-share.ravel())[:args.scan_top]
    subset = sorted((ctx.FL[int(i // ctx.H)], int(i % ctx.H)) for i in order)
    (out / f'scan_{ctx.tag}_subset.json').write_text(json.dumps(subset))
    # phase 2: single-head patching on translation pairs
    for c in nat:
        for f in ('point', 'bbox'):
            gid = f"{c['id']}:{f}"
            g = gens.get(gid)
            if gid in finished or g is None or g['values'] is None:
                continue
            t0 = time.time()
            img_a, _ = e2.render(c, ctx.r3)
            img_b, _ = e2.render(c, ctx.r3, 'B')
            prompt = eh.prompt_for(c['expr'], f)
            pa, pb = ctx.prep(img_a, prompt), ctx.prep(img_b, prompt)
            rec = scan_pair(ctx, pa, pb, g['ids'], g['values'], subset, extra_rows={})
            rec.update({'id': gid, 'case': c['id'], 'format': f, 'seconds': round(time.time() - t0, 1)})
            append_jsonl(path, [rec])
            print(f'vlscan {gid} {rec["seconds"]}s', flush=True)


def scan_pair(ctx: Ctx, pa, pb, ans, vals, singles, extra_rows: dict, kv_layers=()):
    """Teacher-forced run A / run B and head-output patching rows: none, all heads, every single head of `singles`,
    plus extra_rows {name: callable(ca, cb) -> ops}. Returns the record with first-digit logit differences."""
    rt, args = ctx.rt, ctx.args
    L = len(pa.ids)
    start = L - 1
    xa, pos = cr.teacher_forced(rt, pa, ans[:-1])
    xb, _ = cr.teacher_forced(rt, pb, ans[:-1])
    first = [L - 1 + v['toks'][0] for v in vals]
    span = span_of(pa)
    dsa, dsb = ctx.ds(pa), ctx.ds(pb)
    with torch.inference_mode():
        with TailCapture(rt, start, ctx.FL, (), kv_layers, span) as ca:
            forward_logits(rt, xa, pos, first, None if dsa is None else [dsa], span)
        with TailCapture(rt, start, ctx.FL, (), kv_layers, span) as cb:
            lg_b = forward_logits(rt, xb, pos, first, None if dsb is None else [dsb], span)
    hd = ca.o[ctx.FL[0]].shape[-1] // ctx.H
    hb = lambda l, h: cb.o[l][:, h * hd:(h + 1) * hd]
    names, rows = ['none', 'all_heads'], [[], [('head', 0, l, h, hb(l, h)) for l, h in ctx.all_heads]]
    for n, fn in extra_rows.items():
        names.append(n)
        rows.append(fn(ca, cb, hb))
    for l, h in singles:
        names.append(hname(l, h))
        rows.append([('head', 0, l, h, hb(l, h))])
    with torch.inference_mode():
        lg = batched_rows(rt, xa, pos, rows, start, first, args.scan_batch, dsa, span)
    lds = first_digit_ld(lg, lg_b[0], ans, vals, ctx.digits)
    return {'conds': names, 'lds': lds, 'values': [{k: v[k] for k in ('key', 'value', 'toks')} for v in vals]}


# ----------------------------------------------------------------------------- stage: mix

def mix_rows(ctx: Ctx, regions: dict, s0: int, intern_a=None, intern_b=None) -> dict:
    """Extra condition rows for a mix pair: head groups, region splices of image K/V in every full-attention
    layer, and (Qwen3.5) attention-pattern transplants."""
    g = ctx.groups
    rows = {}
    if ctx.q35:
        ex = g['minx'] + g['maxx']
        ey = g['miny'] + g['maxy']
        rows['H11'] = lambda ca, cb, hb: [('head', 0, *H11, hb(*H11))]
        rows['edges_x'] = lambda ca, cb, hb: [('head', 0, l, h, hb(l, h)) for l, h in ex]
        rows['edges_y'] = lambda ca, cb, hb: [('head', 0, l, h, hb(l, h)) for l, h in ey]
        rows['all_but_H11'] = lambda ca, cb, hb: [('head', 0, l, h, hb(l, h)) for l, h in ctx.all_heads if (l, h) != H11]
        for L_ in (16, 20, 24, 28):
            rows[f'L{L_}'] = lambda ca, cb, hb, L_=L_: [('head', 0, L_, h, hb(L_, h)) for h in range(ctx.H)]
    for name in REGIONS:
        pos = torch.as_tensor(regions[name] + s0, dtype=torch.long)
        if len(pos) == 0:
            continue
        rows[f'kv_{name}'] = lambda ca, cb, hb, pos=pos: [
            ('kv', 0, l, pos.to(cb.k[l].device), (cb.k[l][pos - s0], cb.v[l][pos - s0])) for l in ctx.FL]
    if ctx.q35 and intern_a is not None:
        for n, (a_set, b_set) in transplant_sets(ctx, intern_a, intern_b).items():
            rows[n + '_A'] = lambda ca, cb, hb, a_set=a_set: a_set
            rows[n + '_B'] = lambda ca, cb, hb, b_set=b_set: b_set
    return rows


EDGE_CAUSAL = [(20, 0), (24, 2), (24, 7), (24, 12), (24, 4), (20, 2), (20, 10), (24, 5), (24, 9), (28, 0)]


def transplant_sets(ctx: Ctx, ia: dict, ib: dict) -> dict:
    """Pattern transplants (Qwen3.5). For each run R in {A, B}:
      edges_h11pat: every causal edge head keeps its own values and gate but uses L20 H11's attention probabilities;
      h11_edgepat:  L20 H11 keeps its values and gate but uses the mean attention probabilities of the edge heads;
      edges_ownpat_recomputed: edge heads recomputed with their own pattern (checks the recomputation).
    Row 'X_A' sets the heads to the run-A version, 'X_B' to the run-B version, both inside run A; the recovery of
    X is computed from the difference of the two rows."""
    out = {}
    for name in ('edges_h11pat', 'h11_edgepat'):
        sets = []
        for it in (ia, ib):
            if name == 'edges_h11pat':
                pat = it[20]['a'][11]
                sets.append([('head', 0, l, h, pattern_output(it, l, h, pat)) for l, h in EDGE_CAUSAL])
            else:
                pat = torch.stack([it[l]['a'][h] for l, h in EDGE_CAUSAL]).mean(0)
                sets.append([('head', 0, 20, 11, pattern_output(it, 20, 11, pat))])
        out[name] = tuple(sets)
    return out


def stage_mix(ctx: Ctx) -> None:
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / f'mix_{ctx.tag}.jsonl'
    mdir = out / f'mix_{ctx.tag}'
    mdir.mkdir(exist_ok=True)
    finished = done_ids(path)
    subset = None
    if not ctx.q35:
        subset = [tuple(h) for h in json.loads((out / f'scan_{ctx.tag}_subset.json').read_text())]
    items = [c for c in ctx.mix.values() if not ctx.args.mix_kinds or c['kind'] in ctx.args.mix_kinds.split(',')]
    if ctx.args.mix_per_kind:
        cnt: dict = {}
        keep = []
        for c in items:
            key = (c['kind'], c.get('ndistr'))
            cnt[key] = cnt.get(key, 0) + 1
            if cnt[key] <= ctx.args.mix_per_kind:
                keep.append(c)
        items = keep
    if ctx.args.limit:
        items = [c for i, c in enumerate(items) if i % max(1, len(items) // ctx.args.limit) == 0][:ctx.args.limit]
    for c in items:
        if c['id'] in finished:
            continue
        t0 = time.time()
        img_a, box_a = render_mix(c, ctx.r3, False)
        img_b, box_b = render_mix(c, ctx.r3, True)
        prompt = eh.prompt_for(c['expr'], 'point')
        pa, pb = ctx.prep(img_a, prompt), ctx.prep(img_b, prompt)
        if len(pa.ids) != len(pb.ids):
            raise ValueError(f"Prompt lengths differ for {c['id']}")
        # greedy answers of run A and run B
        X = torch.cat([cr.embeds(rt, pa), cr.embeds(rt, pb)], 0)
        P = pa.pos.expand(-1, 2, -1).contiguous()
        with torch.inference_mode(), cv.deepstack_rows(rt, None if ctx.q35 else [ctx.ds(pa), ctx.ds(pb)], span_of(pa)):
            res = cr.greedy(rt, X, P, args.max_new_tokens)
        vals = cf.answer_values(rt, res[0]['ids'], 'point')
        rec = {'id': c['id'], 'kind': c['kind'], 'ndistr': c.get('ndistr'), 'box_a': box_a, 'box_b': box_b,
               'canvas': c['canvas'], 'grid': list(pa.grid), 'text_a': res[0]['text'], 'text_b': res[1]['text'],
               'values_b': value_rows(rt, [res[1]], 'point')[0]}
        if vals is None:
            rec['skipped'] = 'run A answer not parseable'
            append_jsonl(path, [rec])
            continue
        L = len(pa.ids)
        ans = res[0]['ids']
        xa, pos = cr.teacher_forced(rt, pa, ans[:-1])
        xb, _ = cr.teacher_forced(rt, pb, ans[:-1])
        first = [L - 1 + v['toks'][0] for v in vals]
        with torch.inference_mode():
            att_a = attention_at(ctx, xa, pos, first, pa)                   # (nL, H, 2, Nv)
            att_b = attention_at(ctx, xb, pos, first, pb)
            ia = ib = None
            if ctx.q35:
                ia = head_internals(rt, xa, pos, (20, 24, 28), L - 1)
                ib = head_internals(rt, xb, pos, (20, 24, 28), L - 1)
                rec['recompute_err'] = max(float((d['o'] - torch.sigmoid(d['gate']) * torch.einsum(
                    'htl,hld->thd', d['a'], d['v'].repeat_interleave(d['groups'], 0))).abs().max() / d['o'].abs().max())
                    for it in (ia, ib) for d in it.values())
        regions = token_regions(box_a, box_b, pa.grid)
        extra = mix_rows(ctx, regions, pa.span['start'], ia, ib)
        singles = ctx.all_heads if ctx.q35 else subset
        sc = scan_pair(ctx, pa, pb, ans, vals, singles, extra, kv_layers=ctx.FL)
        rec.update(sc)
        rec['region_sizes'] = {k: int(len(v)) for k, v in regions.items()}
        keep = [i for i, l in enumerate(ctx.FL) if l >= 12]
        if ctx.q35:
            arr_a, arr_b = att_a[keep], att_b[keep]
        else:
            idx = np.array([[ctx.FL.index(l), h] for l, h in subset])
            arr_a, arr_b = att_a[idx[:, 0], idx[:, 1]], att_b[idx[:, 0], idx[:, 1]]
            rec['image_share_all'] = att_a.sum(-1).round(4).tolist()
        np.savez_compressed(mdir / f"{c['id']}.npz", att_a=arr_a.astype(np.float16), att_b=arr_b.astype(np.float16))
        rec['seconds'] = round(time.time() - t0, 1)
        append_jsonl(path, [rec])
        del ia, ib
        print(f"mix {c['id']} {c['kind']} {rec['seconds']}s", flush=True)


# ----------------------------------------------------------------------------- stage: mix2 (per-head transplants)

TRANSPLANT_HEADS = [(24, 5), (20, 10), (24, 7), (20, 0), (28, 0), (24, 2), (24, 9), (24, 12)]


def stage_mix2(ctx: Ctx) -> None:
    """Qwen3.5, every mix pair: per-head attention-pattern transplants between L20 H11 and single edge heads.
    Rows (each as run-A version and run-B version inside run A, recovery from the difference):
      own:<h>       head h recomputed with its own pattern (checks the recomputation, equals single-head patching);
      h11pat:<h>    head h with L20 H11's attention probabilities and its own values / gate;
      pat_of:<h>    L20 H11 with head h's attention probabilities and H11's values / gate;
      edges10_own   the ten causal edge heads with their own patterns (baseline of edges_h11pat in stage mix)."""
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / f'mix2_{ctx.tag}.jsonl'
    finished = done_ids(path)
    gens = {r['id']: r for r in read_jsonl(out / f'mix_{ctx.tag}.jsonl') if 'lds' in r}
    for c in ctx.mix.values():
        g = gens.get(c['id'])
        if c['id'] in finished or g is None:
            continue
        t0 = time.time()
        img_a, box_a = render_mix(c, ctx.r3, False)
        img_b, box_b = render_mix(c, ctx.r3, True)
        prompt = eh.prompt_for(c['expr'], 'point')
        pa, pb = ctx.prep(img_a, prompt), ctx.prep(img_b, prompt)
        L = len(pa.ids)
        ans = rt.tokenizer.encode(g['text_a'], add_special_tokens=False)
        vals = cf.answer_values(rt, ans, 'point')
        if vals is None or [v['value'] for v in vals] != [v['value'] for v in g['values']]:
            append_jsonl(path, [{'id': c['id'], 'skipped': 'answer re-tokenisation differs'}])
            continue
        ans = ans + [rt.tokenizer.convert_tokens_to_ids('<|im_end|>')]
        xa, pos = cr.teacher_forced(rt, pa, ans[:-1])
        xb, _ = cr.teacher_forced(rt, pb, ans[:-1])
        first = [L - 1 + v['toks'][0] for v in vals]
        with torch.inference_mode():
            lg_b = forward_logits(rt, xb, pos, first)
            ia = head_internals(rt, xa, pos, (20, 24, 28), L - 1)
            ib = head_internals(rt, xb, pos, (20, 24, 28), L - 1)
        names, rows = ['none'], [[]]
        for (l, h) in TRANSPLANT_HEADS:
            for tag, mk in (('own', lambda it, l=l, h=h: [('head', 0, l, h, pattern_output(it, l, h, it[l]['a'][h]))]),
                            ('h11pat', lambda it, l=l, h=h: [('head', 0, l, h, pattern_output(it, l, h, it[20]['a'][11]))]),
                            ('pat_of', lambda it, l=l, h=h: [('head', 0, 20, 11, pattern_output(it, 20, 11, it[l]['a'][h]))])):
                for run, it in (('A', ia), ('B', ib)):
                    names.append(f'{tag}:{hname(l, h)}_{run}')
                    rows.append(mk(it))
        for run, it in (('A', ia), ('B', ib)):
            names.append(f'edges10_own_{run}')
            rows.append([('head', 0, l, h, pattern_output(it, l, h, it[l]['a'][h])) for l, h in EDGE_CAUSAL])
        with torch.inference_mode():
            lg = batched_rows(rt, xa, pos, rows, L - 1, first, args.scan_batch)
        lds = first_digit_ld(lg, lg_b[0], ans, vals, ctx.digits)
        append_jsonl(path, [{'id': c['id'], 'kind': c['kind'], 'conds': names, 'lds': lds, 'seconds': round(time.time() - t0, 1)}])
        print(f"mix2 {c['id']} {time.time() - t0:.1f}s", flush=True)
        del ia, ib


# ----------------------------------------------------------------------------- stage: spot

@torch.inference_mode()
def stage_spot(ctx: Ctx) -> None:
    """Answer agreement of row A of the batched decoder (avg records) with qwen_runtime.generate."""
    rt, out, args = ctx.rt, ctx.out, ctx.args
    path = out / f'spot_{ctx.tag}.jsonl'
    finished = done_ids(path)
    recs = read_jsonl(out / f'avg_{ctx.tag}.jsonl')
    rng = np.random.default_rng(args.seed)
    for k in sorted(rng.choice(len(recs), min(20, len(recs)), replace=False)):
        r = recs[int(k)]
        if r['id'] in finished:
            continue
        c = ctx.r2[r['case']]
        img, _ = e2.render(c, ctx.r3)
        ref = qr.generate(rt, qr.build(rt, [img], eh.prompt_for(c['expr'], r['format'])), args.max_new_tokens)
        vr = cf.answer_values(rt, rt.tokenizer.encode(ref['text'], add_special_tokens=False), r['format'])
        vr = None if vr is None else {d['key']: d['value'] for d in vr}
        append_jsonl(path, [{'id': r['id'], 'reference': ref['text'], 'batched': r['gen'][0]['text'],
                             'batch_rows': len(r['gen']), 'answer_equal': vr == r['gen'][0]['values']}])


# ----------------------------------------------------------------------------- main

STAGES = {'avg': stage_avg, 'vlscan': stage_vlscan, 'mix': stage_mix, 'mix2': stage_mix2, 'spot': stage_spot}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', required=True, choices=['cases', *STAGES])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--round3', type=Path, required=True)
    p.add_argument('--round4', type=Path, required=True)
    for k in ('minx', 'maxx', 'miny', 'maxy'):
        p.add_argument(f'--group-{k}', default='')
    p.add_argument('--n-mix', type=int, default=16, help='mix items per kind and distractor count')
    p.add_argument('--n-grid', type=int, default=32)
    p.add_argument('--scan-top', type=int, default=128, help='Qwen3-VL: heads with the largest image share to scan')
    p.add_argument('--cond-batch', type=int, default=12)
    p.add_argument('--scan-batch', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=40)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--max-memory', default='')
    p.add_argument('--mix-kinds', default='', help='comma-separated mix kinds to run (default: all)')
    p.add_argument('--mix-per-kind', type=int, default=0, help='at most this many mix items per kind and distractor count')
    p.add_argument('--limit', type=int, default=0, help='smoke test: process only this many items per stage')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = list(args.stages)
    if 'cases' in stages:
        stage_cases(args.output, args)
        stages.remove('cases')
    if not stages:
        return
    rt = cf.load_runtime(args.model, args.max_memory)
    ctx = Ctx(rt, args.output, args)
    for st in stages:
        meta = {'stage': st, 'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt),
                'code_sha256': e2.sha256_file(Path(__file__)), 'started': datetime.now(timezone.utc).isoformat(),
                'device': torch.cuda.get_device_name(0)}
        append_jsonl(args.output / 'run_log.jsonl', [meta])
        STAGES[st](ctx)


if __name__ == '__main__':
    main()
