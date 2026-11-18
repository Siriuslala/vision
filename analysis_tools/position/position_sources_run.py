"""Experiment 3.15 round 3 GPU stages: sources of position information (Qwen3.5-9B and Qwen3-VL-8B).

Results: docs/analysis/position/results_2026-11-11_position_sources.md. bf16, SDPA, greedy, thinking off.

Stages
  capture  (Qwen3.5) one prefill per image of three families (jgrid: decorrelated grids; single: one red figure on
           a blank canvas; refcoco: RefCOCO test photos) under two prefill conditions (clean; 'nof' = no forgetting,
           alpha = 1 at image tokens of the DeltaNet layers 9-15). At sampled image tokens it stores: the raw Gated
           DeltaNet head outputs o_t = S_t^T q_t of selected layers, the residual stream (decoder input and outputs
           of layers 8, 16, 19), the DeltaNet mixer outputs of layers 9-15 (residual-space contributions), alpha and
           beta of every DeltaNet head, and for layers 9-15 the exact read-out weights A[n, m] of every head
           (o_n = sum_m A[n, m] v_m) summarised per query token, plus the split of o_n into an 'amount' part
           (sum_m A[n, m]) * mean(v) and a 'content' part sum_m A[n, m] (v_m - mean(v)). Also the direct point
           answers (and row/column answers on jgrid) under both conditions.
  roll     (both models) raster-order dissociation: the image tokens are written into the image span in a rolled
           order (rows cyclically shifted, or tokens inside every row cyclically shifted), each token keeping its
           own embedding and MRoPE ids (and, for Qwen3-VL, its DeepStack features). For Qwen3.5 additional rows
           restore the DeltaNet mixer outputs at image slots to their unrolled values (all DeltaNet layers or
           layers 9-15). Families grid10 / single / refcoco; point answers, and row/column answers on grid10.
           Also (Qwen3.5) a layer-band sweep of the no-forgetting edit (layers 1-7, 9-15, 17-23, 25-31).
  steer    (Qwen3.5) adds fitted directions (analysis output steer_dirs.pt) to the DeltaNet mixer outputs of layers
           9-15 at all image tokens: unique effect of +1 entity row started ('cnt'), of +2 token rows of own
           position ('pos'), and the column analogues; point and row/column answers.
  vlflow   (Qwen3-VL) activation patching per answer digit on (A, B) pairs differing only in target position
           (grid10 neighbour cells, RefCOCO crop-shift pairs): image KV at the studied step / over a window,
           earlier answer-token KV, head-output swaps by layer bin and for selected heads, B-image visibility scan,
           and the image attention share of every head at each value's hundreds step.

Layers 1-based; heads 0-based.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.behavior import cot_mechanism as cm
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import causal_vs_linear as cv
from analysis_tools.position import coordinate_mechanism_natural as cmn
from analysis_tools.position import coordinate_mechanism_natural_run as cmr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import edge_heads as eh
from analysis_tools.position import ordinal_vs_coordinate as ov
from analysis_tools.position import position_sources as ps
from analysis_tools.position import sequence_order as so

Q35_FULL = (4, 8, 12, 16, 20, 24, 28, 32)
Q35_LIN = tuple(l for l in range(1, 33) if l not in Q35_FULL)
NOF_LAYERS = (9, 10, 11, 13, 14, 15)
O_LAYERS = (2, 5, 7, 9, 10, 11, 13, 14, 15, 18, 22, 26, 30)       # raw head outputs stored (clean)
O_LAYERS_NOF = (9, 10, 11, 13, 14, 15, 18, 22)
RES_LAYERS = (0, 8, 16, 19)
BANDS = {'nof1_7': (1, 2, 3, 5, 6, 7), 'nof9_15': NOF_LAYERS, 'nof17_23': (17, 18, 19, 21, 22, 23),
         'nof25_31': (25, 26, 27, 29, 30, 31)}
N_SAMPLE = 32                                   # random image tokens per image (plus target / after-target tokens)
N_QUERY = 24                                    # query tokens per image for the A-weight summaries
N_AMAP = 4                                      # images per family whose full A rows (layers 9-15) are stored


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def code_digest() -> dict:
    here = Path(__file__).parent
    return {p: hashlib.sha256((here / p).read_bytes()).hexdigest()[:16]
            for p in ('position_sources.py', 'position_sources_run.py')}


def point_prompt(target: str) -> str:
    return eh.prompt_for(target, 'point')


ROWCOL_PROMPT = cm.prompt('rowcol', 'direct')


# ----------------------------------------------------------------------------- cases

def family_cases(fam: str, n: int, args) -> list[dict]:
    if fam == 'jgrid':
        return ps.jgrid_cases(n)
    if fam == 'single':
        return cmn.single_cases(n)
    if fam == 'grid10':
        return cmn.grid_cases('grid10', n)
    if fam == 'refcoco':
        a = argparse.Namespace(work=args.work, data_dir=args.data_dir, n_refcoco=n)
        return cmr.Cases(a).refcoco(n)
    raise ValueError(fam)


def render(case: dict, args) -> tuple[Image.Image, dict, str]:
    fam = case['family']
    if fam == 'jgrid':
        img, t = ps.jgrid_image(case)
        return img, t, cmn.SYNTH_TARGET
    a = argparse.Namespace(work=args.work, data_dir=args.data_dir, n_refcoco=1)
    img, t = cmr.Cases(a).render(case)
    return img, t, cmn.target_text(case)


# ----------------------------------------------------------------------------- DeltaNet capture

class DeltaTap:
    """Wraps the chunked Gated DeltaNet kernel to record (q, k, v, g, beta) and the head output per layer.

    `layer` is set by a forward pre-hook of each linear-attention module; only prefill calls (sequence length >
    1) whose batch size is 1 are recorded. Tensors are kept on the GPU in float32 until `pop()`.
    """

    def __init__(self, rt):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        self.mq = mq
        self.orig = mq.torch_chunk_gated_delta_rule
        self.layer = None
        self.store: dict[int, dict] = {}
        self.handles = []
        for l in Q35_LIN:
            def pre(m, args, kwargs, l=l):
                self.layer = l
            self.handles.append(rt.layers[l - 1].linear_attn.register_forward_pre_hook(pre, with_kwargs=True))
        tap = self

        def wrapped(query, key, value, g, beta, **kw):
            out = tap.orig(query, key, value, g=g, beta=beta, **kw)
            if tap.layer is not None and query.shape[0] == 1 and query.shape[1] > 1:
                qn = torch.nn.functional.normalize(query[0].float(), dim=-1, eps=1e-6) / query.shape[-1] ** 0.5
                kn = torch.nn.functional.normalize(key[0].float(), dim=-1, eps=1e-6)
                tap.store[tap.layer] = {'q': qn, 'k': kn, 'v': value[0].float(), 'g': g[0].float(),
                                        'beta': beta[0].float(), 'o': out[0][0].float()}
            return out
        mq.torch_chunk_gated_delta_rule = wrapped

    def close(self):
        self.mq.torch_chunk_gated_delta_rule = self.orig
        for h in self.handles:
            h.remove()


@contextmanager
def no_forget(rt, layers, span: tuple[int, int], prompt_len: int | None = None):
    """alpha = 1 at image-token positions [s, e) in the given DeltaNet layers (prefill passes only)."""
    s, e = span
    hs = []
    for l in layers:
        mod = rt.layers[l - 1].linear_attn

        def hook(m, inp, out, mod=mod):
            if out.shape[1] < e or (prompt_len is not None and out.shape[1] != prompt_len):
                return out
            out = out.clone()
            out[:, s:e] = ov.graded_a(out[:, s:e], mod.dt_bias, 0.0).to(out.dtype)
            return out
        hs.append(mod.in_proj_a.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.inference_mode()
def capture_one(rt, image: Image.Image, prompt: str, table: dict, toks: np.ndarray, queries: np.ndarray,
                cond: str, keep_amap: bool) -> dict:
    batch = qr.build(rt, [image], prompt)
    span = qr.image_spans(rt, batch)[0]
    s, e = span['start'], span['end']
    tap = DeltaTap(rt)
    res, mix = {}, {}
    hs = []
    for l in RES_LAYERS:
        if l == 0:
            def pre(m, args, kwargs):
                h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
                res[0] = h[0, s:e].float()
            hs.append(rt.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
        else:
            def post(m, args, out, l=l):
                res[l] = (out[0] if isinstance(out, tuple) else out)[0, s:e].float()
            hs.append(rt.layers[l - 1].register_forward_hook(post))
    for l in NOF_LAYERS:
        def mpost(m, args, out, l=l):
            mix[l] = (out[0] if isinstance(out, tuple) else out)[0, s:e].float()
        hs.append(rt.layers[l - 1].linear_attn.register_forward_hook(mpost))
    try:
        with ExitStack() as stack:
            if cond == 'nof':
                stack.enter_context(no_forget(rt, NOF_LAYERS, (s, e)))
            rt.model(**batch, use_cache=False, logits_to_keep=1)
    finally:
        tap.close()
        for h in hs:
            h.remove()
    tt = torch.as_tensor(toks, device=rt.device)
    out = {'res': {l: v[tt].half().cpu() for l, v in res.items()},
           'mix': {l: v[tt].half().cpu() for l, v in mix.items()}}
    o_layers = O_LAYERS if cond == 'clean' else O_LAYERS_NOF
    out['o'] = {l: tap.store[l]['o'][s:e][tt].half().cpu() for l in o_layers}              # (S, 32, 128)
    out['alpha'] = torch.stack([tap.store[l]['g'][s:e][tt].exp() for l in Q35_LIN]).half().cpu()  # (24, S, 32)
    out['beta'] = torch.stack([tap.store[l]['beta'][s:e][tt] for l in Q35_LIN]).half().cpu()
    # A-weight summaries (layers 9-15) at query tokens
    H, W = span['grid']
    qabs = queries + s
    ii = torch.as_tensor(table['i'], device=rt.device, dtype=torch.float32)
    jj = torch.as_tensor(table['j'], device=rt.device, dtype=torch.float32)
    ent = torch.as_tensor(table.get('ent', np.zeros(H * W)), device=rt.device, dtype=torch.float32)
    tgt = torch.as_tensor(table['tgt'], device=rt.device, dtype=torch.float32)
    nn_ = torch.arange(H * W, device=rt.device, dtype=torch.float32)
    qi = ii[torch.as_tensor(queries, device=rt.device)]
    summ, parts, amap = {}, {}, {}
    for l in NOF_LAYERS:
        d = tap.store[l]
        A = ps.delta_weights(d['q'], d['k'], d['g'], d['beta'], qabs)       # (Q, 32, T)
        Ai = A[:, :, s:e]                                                   # image sources
        a_abs = Ai.abs()
        wsum = a_abs.sum(-1).clamp_min(1e-12)
        w = a_abs / wsum[..., None]
        v_img = d['v'][s:e]                                                 # (N, 32, 128)
        vbar = v_img.mean(0)                                                # (32, 128)
        o_q = d['o'][qabs]                                                  # (Q, 32, 128)
        amount = Ai.sum(-1)                                                 # (Q, 32)
        o_img = torch.einsum('qhm,mhd->qhd', Ai, v_img)
        o_all = torch.einsum('qhm,mhd->qhd', A, d['v'][:A.shape[-1]])
        o_pre = o_all - o_img                                               # contribution of text before image
        content = torch.einsum('qhm,mhd->qhd', Ai, v_img - vbar[None])
        summ[l] = {
            'amount': amount, 'abs_mass': a_abs.sum(-1),
            'src_i': (w * ii).sum(-1), 'src_j': (w * jj).sum(-1),
            'lag': (w * (torch.as_tensor(queries, device=rt.device, dtype=torch.float32)[:, None, None] - nn_)).sum(-1),
            'same_row': (w * (ii[None, None, :] == qi[:, None, None]).float()).sum(-1),
            'ent': (w * ent).sum(-1), 'tgt': (w * tgt).sum(-1),
            'pre_share': o_pre.norm(dim=-1) / o_q.norm(dim=-1).clamp_min(1e-12),
            'recon_err': (o_all - o_q).norm(dim=-1) / o_q.norm(dim=-1).clamp_min(1e-12),
        }
        summ[l] = {k: v.half().cpu() for k, v in summ[l].items()}
        parts[l] = {'amount_part': (amount[..., None] * vbar[None]).half().cpu(), 'content_part': content.half().cpu(),
                    'o': o_q.half().cpu()}
        if keep_amap:
            amap[l] = Ai.half().cpu()
    out['summ'], out['parts'] = summ, parts
    if keep_amap:
        out['amap'] = amap
    return out


def stage_capture(rt, args) -> None:
    out_dir = args.output / 'capture'
    out_dir.mkdir(exist_ok=True)
    meta_path = out_dir / 'meta.jsonl'
    done = {(r['family'], r['case'], r['cond']) for r in read_jsonl(meta_path)}
    fams = {'jgrid': args.n_jgrid, 'single': args.n_single, 'refcoco': args.n_refcoco}
    t0 = time.time()
    for fam, n in fams.items():
        cases = family_cases(fam, n, args)
        for ci, case in enumerate(cases):
            img, truth, target = render(case, args)
            H, W = img.height // ps.TOKEN, img.width // ps.TOKEN
            table = ps.token_table(truth, H, W, fam)
            rng = ps.rng_for('sample', case['id'])
            toks = ps.sample_tokens(table, N_SAMPLE, rng)
            queries = np.unique(np.concatenate([np.where(table['tgt'] > 0)[0][:4],
                                                np.where(table['after_tgt_row'] > 0)[0][:4],
                                                rng.choice(H * W, N_QUERY - 8, replace=False)]))
            for cond in ('clean', 'nof'):
                if (fam, case['id'], cond) in done:
                    continue
                rec = capture_one(rt, img, point_prompt(target), table, toks, queries, cond,
                                  keep_amap=ci < N_AMAP and cond == 'clean')
                rec['toks'], rec['queries'] = toks, queries
                rec['table'] = {k: v for k, v in table.items()}
                torch.save(rec, out_dir / f"{case['id']}_{cond}.pt")
                append_jsonl(meta_path, [{'family': fam, 'case': case['id'], 'cond': cond, 'H': H, 'W': W,
                                          'truth': {k: v for k, v in truth.items() if k != 'entities'}}])
            print(f'capture {fam} {ci + 1}/{len(cases)} {time.time() - t0:.0f}s', flush=True)
    # answers under both conditions (batched greedy)
    ans_path = args.output / 'capture_answers.jsonl'
    adone = {(r['case'], r['cond'], r['fmt']) for r in read_jsonl(ans_path)}
    for fam, n in fams.items():
        cases = family_cases(fam, n, args)
        fmts = ('point', 'rowcol') if fam == 'jgrid' else ('point',)
        for fmt in fmts:
            for cond in ('clean', 'nof'):
                todo = [c for c in cases if (c['id'], cond, fmt) not in adone]
                for a in range(0, len(todo), args.batch):
                    chunk = todo[a:a + args.batch]
                    items = []
                    for c in chunk:
                        img, truth, target = render(c, args)
                        items.append((img, point_prompt(target) if fmt == 'point' else ROWCOL_PROMPT, truth))
                    res = cmr.gen_batch(rt, [(im, p) for im, p, _ in items], 40, s=0.0 if cond == 'nof' else 1.0)
                    append_jsonl(ans_path, [{'family': fam, 'case': c['id'], 'cond': cond, 'fmt': fmt,
                                             'text': r['text'], 'finish_reason': r['finish_reason'],
                                             'truth': {k: v for k, v in it[2].items() if k != 'entities'},
                                             'batch_size': len(chunk)}
                                            for c, r, it in zip(chunk, res, items)])
                print(f'answers {fam} {fmt} {cond} {time.time() - t0:.0f}s', flush=True)


# NOTE: cmr.gen_batch applies alpha^s at layers 9-16 DeltaNet (9,10,11,13,14,15) image tokens, the same set as
# NOF_LAYERS; s = 0 is the no-forgetting condition.


# ----------------------------------------------------------------------------- roll

def roll_conditions(rt, case_id: str, truth: dict, H: int, W: int) -> list[dict]:
    rng = ps.rng_for('roll', case_id)
    conds = [{'name': 'id', 'order': so.order_identity(H, W), 'k': 0, 'axis': None}]
    kr = ps.roll_shift(truth['box'], H, W, 0, rng)
    kc = ps.roll_shift(truth['box'], H, W, 1, rng)
    if kr is not None:
        conds.append({'name': 'rowroll', 'order': so.order_roll_rows(H, W, kr), 'k': kr, 'axis': 0})
    if kc is not None:
        conds.append({'name': 'colroll', 'order': so.order_roll_within_rows(H, W, kc), 'k': kc, 'axis': 1})
    if rt.family == 'qwen3_5':
        extra = []
        for c in conds[1:]:
            extra.append(dict(c, name=c['name'] + '_rDN', restore=Q35_LIN))
            if c['axis'] == 0:
                extra.append(dict(c, name=c['name'] + '_rDN9', restore=NOF_LAYERS))
                extra.append(dict(c, name=c['name'] + '_rDN1_7', restore=BANDS['nof1_7']))
        conds += extra
    return conds


@torch.inference_mode()
def stage_roll(rt, args) -> None:
    tag = 'vl' if rt.family == 'qwen3_vl' else 'q35'
    path = args.output / f'roll_{tag}.jsonl'
    done = {(r['case'], r['fmt'], r['cond']) for r in read_jsonl(path)}
    t0 = time.time()
    for fam, n in (('grid10', args.n_roll), ('single', args.n_roll), ('refcoco', args.n_roll)):
        cases = family_cases(fam, n, args)
        for ci, case in enumerate(cases):
            img, truth, target = render(case, args)
            fmts = ('point', 'rowcol') if fam == 'grid10' else ('point',)
            for fmt in fmts:
                prompt = point_prompt(target) if fmt == 'point' else ROWCOL_PROMPT
                prep = cv.prepare(rt, img, prompt)
                H, W = prep.grid
                conds = roll_conditions(rt, case['id'], truth, H, W)
                if rt.family == 'qwen3_vl':
                    conds += [dict(c, name=c['name'] + '_dsoff', ds=False) for c in conds if c['name'] in ('id', 'rowroll')]
                elif fmt == 'point':
                    conds += [{'name': b, 'order': so.order_identity(H, W), 'k': 0, 'axis': None, 'band': BANDS[b]}
                              for b in BANDS]
                conds = [c for c in conds if (case['id'], fmt, c['name']) not in done]
                if not conds:
                    continue
                span = (prep.span['start'], prep.span['end'])
                clean = so.capture_mixers(rt, *so.build(rt, prep, so.permuted(prep, so.order_identity(H, W))), span) \
                    if any(c.get('restore') for c in conds) else None
                recs = []
                for a in range(0, len(conds), args.batch):
                    chunk = conds[a:a + args.batch]
                    seqs = [so.permuted(prep, c['order']) for c in chunk]
                    built = [so.build(rt, prep, sq) for sq in seqs]
                    x = torch.cat([b[0] for b in built], 0)
                    pos = torch.cat([b[1] for b in built], 1)
                    with ExitStack() as stack:
                        if rt.family == 'qwen3_vl':
                            ds = prep.meta['ds']
                            rows = []
                            for c in chunk:
                                o = torch.as_tensor(c['order'], device=ds[0].device)
                                rows.append([d[o] if c.get('ds', True) else torch.zeros_like(d) for d in ds])
                            stack.enter_context(cv.deepstack_rows(rt, rows, span))
                        else:
                            rr = [(c['order'], c.get('restore', ()), None) for c in chunk]
                            if any(r[1] for r in rr):
                                stack.enter_context(so.restore_mixers(rt, clean, span, rr, x.shape[1]))
                            for bi, c in enumerate(chunk):
                                if c.get('band'):
                                    stack.enter_context(row_no_forget(rt, c['band'], span, bi, x.shape[1]))
                        res = cr.greedy(rt, x, pos, 40)
                    for c, r in zip(chunk, res):
                        rec = {'family': fam, 'case': case['id'], 'fmt': fmt, 'cond': c['name'], 'k': c['k'],
                               'axis': c['axis'], 'H': H, 'W': W, 'text': r['text'],
                               'finish_reason': r['finish_reason'], 'batch_size': len(chunk),
                               'truth': {k: v for k, v in truth.items() if k != 'entities'}}
                        recs.append(rec)
                append_jsonl(path, recs)
                del clean
            print(f'roll {tag} {fam} {ci + 1}/{len(cases)} {time.time() - t0:.0f}s', flush=True)


@contextmanager
def row_no_forget(rt, layers, span: tuple[int, int], row: int, prompt_len: int):
    """alpha = 1 at image tokens of one batch row in the given DeltaNet layers (prefill only)."""
    s, e = span
    hs = []
    for l in layers:
        mod = rt.layers[l - 1].linear_attn

        def hook(m, inp, out, mod=mod):
            if out.shape[1] != prompt_len:
                return out
            out = out.clone()
            out[row, s:e] = ov.graded_a(out[row, s:e], mod.dt_bias, 0.0).to(out.dtype)
            return out
        hs.append(mod.in_proj_a.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


# ----------------------------------------------------------------------------- steer

@contextmanager
def add_dirs(rt, dirs: dict[int, torch.Tensor], span: tuple[int, int], rows: list[float], prompt_len: int):
    """Add rows[b] * dirs[l] to the DeltaNet mixer output of layer l at image tokens of batch row b (prefill)."""
    s, e = span
    hs = []
    scale = torch.tensor(rows, device=rt.device)
    for l, d in dirs.items():
        def hook(m, args, out, d=d):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] != prompt_len:
                return out
            h = h.clone()
            h[:, s:e] += (scale[:, None, None] * d.to(h)[None, None, :]).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hs.append(rt.layers[l - 1].linear_attn.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.inference_mode()
def stage_steer(rt, args) -> None:
    dirs_all = torch.load(args.output / 'steer_dirs.pt')
    path = args.output / 'steer.jsonl'
    done = {(r['case'], r['fmt'], r['cond']) for r in read_jsonl(path)}
    t0 = time.time()
    plan = []
    for name in dirs_all['names']:
        for sgn in dirs_all['scales']:
            plan.append((f'{name}{sgn:+g}', name, sgn))
    for fam, n in (('jgrid', args.n_steer), ('single', args.n_steer // 2), ('refcoco', args.n_steer // 2)):
        cases = family_cases(fam, n, args)
        if fam == 'jgrid':
            cases = cases[-n:]                       # steering images are the last jgrid layouts (probe fit on first)
        for ci, case in enumerate(cases):
            img, truth, target = render(case, args)
            for fmt in (('point', 'rowcol') if fam == 'jgrid' else ('point',)):
                prompt = point_prompt(target) if fmt == 'point' else ROWCOL_PROMPT
                conds = [('clean', None, 0.0)] + plan
                conds = [c for c in conds if (case['id'], fmt, c[0]) not in done]
                if not conds:
                    continue
                prep = cv.prepare(rt, img, prompt)
                span = (prep.span['start'], prep.span['end'])
                recs = []
                for a in range(0, len(conds), args.batch):
                    chunk = conds[a:a + args.batch]
                    x = torch.cat([cr.embeds(rt, prep)] * len(chunk), 0)
                    pos = torch.cat([prep.pos] * len(chunk), 1)
                    # one hook per direction name, with per-row scales (0 for rows using another direction)
                    with ExitStack() as stack:
                        for name in dirs_all['names']:
                            rows = [c[2] if c[1] == name else 0.0 for c in chunk]
                            if any(rows):
                                stack.enter_context(add_dirs(rt, dirs_all['dirs'][name], span, rows, x.shape[1]))
                        res = cr.greedy(rt, x, pos, 40)
                    for c, r in zip(chunk, res):
                        recs.append({'family': fam, 'case': case['id'], 'fmt': fmt, 'cond': c[0], 'dir': c[1],
                                     'scale': c[2], 'text': r['text'], 'finish_reason': r['finish_reason'],
                                     'batch_size': len(chunk),
                                     'truth': {k: v for k, v in truth.items() if k != 'entities'}})
                append_jsonl(path, recs)
            print(f'steer {fam} {ci + 1}/{len(cases)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- Qwen3-VL flow

class VLHeadSwap:
    """o_proj-input head slices of chosen (layer, head) in row r overwritten by the source row (one forward)."""

    def __init__(self, rt, ops: list[tuple[int, int, int]], src: int):
        hd = rt.model.config.text_config.head_dim
        self.handles = []
        by: dict[int, list] = {}
        for r, l, h in ops:
            by.setdefault(l, []).append((r, h))
        for l, rh in by.items():
            def pre(m, args, rh=rh):
                x = args[0].clone()
                for r, h in rh:
                    x[r, :, h * hd:(h + 1) * hd] = x[src, :, h * hd:(h + 1) * hd]
                return (x,) + tuple(args[1:])
            self.handles.append(rt.layers[l - 1].self_attn.o_proj.register_forward_pre_hook(pre))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def vl_copy_kv(cache, dst: int, src: int, positions: list[int], layers) -> None:
    if not positions:
        return
    for l in layers:
        lay = cache.layers[l - 1]
        p = torch.tensor(positions, device=lay.keys.device)
        lay.keys[dst, :, p] = lay.keys[src, :, p]
        lay.values[dst, :, p] = lay.values[src, :, p]


VL_BINS = tuple(tuple(range(a, a + 4)) for a in range(1, 37, 4))      # 9 bins of 4 layers


def vl_conditions(heads: list[tuple[int, int]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    step = ('A', 'B', 'img', 'ans_kv', 'img+ans_kv') + tuple(f'img@{b[0]}-{b[-1]}' for b in VL_BINS)
    win = ('A', 'B', 'img') + tuple(f'heads@{b[0]}-{b[-1]}' for b in VL_BINS) + ('allheads',) + \
        tuple(f'h{l}.{h}' for l, h in heads) + ('topheads',)
    return step, win


@torch.inference_mode()
def vl_run(seq, tau: int, w0: int, conds: tuple[str, ...], heads: list[tuple[int, int]]) -> dict:
    rt = seq.rt
    L = rt.model.config.text_config.num_hidden_layers
    nh = rt.model.config.text_config.num_attention_heads
    rows = [1 if c == 'B' else 0 for c in conds]
    cache, delta = seq.prefill(w0, rows)
    b = conds.index('B')
    img = list(range(*seq.img))
    ans = list(range(seq.P - 1, w0))
    ops = []
    for r, c in enumerate(conds):
        for part in c.split('+'):
            if part in ('A', 'B'):
                continue
            if part == 'img':
                vl_copy_kv(cache, r, b, img, range(1, L + 1))
            elif part.startswith('img@'):
                lo, hi = (int(v) for v in part[4:].split('-'))
                vl_copy_kv(cache, r, b, img, range(lo, hi + 1))
            elif part == 'ans_kv':
                vl_copy_kv(cache, r, b, ans, range(1, L + 1))
            elif part.startswith('heads@'):
                lo, hi = (int(v) for v in part[6:].split('-'))
                ops += [(r, l, h) for l in range(lo, hi + 1) for h in range(nh)]
            elif part == 'allheads':
                ops += [(r, l, h) for l in range(1, L + 1) for h in range(nh)]
            elif part == 'topheads':
                ops += [(r, l, h) for l, h in heads]
            elif part.startswith('h'):
                l, h = (int(v) for v in part[1:].split('.'))
                ops.append((r, l, h))
            else:
                raise ValueError(part)
    with VLHeadSwap(rt, ops, b):
        logits = seq.forward(cache, delta, rows, w0, tau)
    del cache
    return {'conds': list(conds), 'lp': cmr.digit_lp(rt, logits).round(4).tolist()}


@torch.inference_mode()
def vl_tail(seq, tau: int, first: int) -> dict:
    rt = seq.rt
    L = rt.model.config.text_config.num_hidden_layers
    out = []
    starts = sorted({min(first + o, tau) for o in cmr.TAIL_OFFSETS})
    for st_ in starts:
        cache, delta = seq.prefill(st_, [0, 1, 0])
        vl_copy_kv(cache, 2, 1, list(range(*seq.img)), range(1, L + 1))
        out.append(cmr.digit_lp(rt, seq.forward(cache, delta, [0, 1, 0], st_, tau)).round(4).tolist())
        del cache
    return {'starts': [s_ - first for s_ in starts], 'lp': out}


@torch.inference_mode()
def vl_attention(seq, steps: list[int], box_tokens: list[int]) -> dict:
    """Image attention share and in-target share of every head at the given fed positions (row A only)."""
    rt = seq.rt
    s0, s1 = seq.img
    cache, delta = seq.prefill(min(steps), [0])
    try:
        ids = seq.ids[:1, min(steps):max(steps) + 1]
        ar = torch.arange(min(steps), max(steps) + 1, device=ids.device)
        pos = (ar[None, None, :] + delta.view(1, 1, 1)).expand(3, 1, -1).contiguous()
        rt.model.config._attn_implementation = 'eager'
        for m in rt.model.modules():
            if hasattr(m, 'config') and hasattr(m.config, '_attn_implementation'):
                m.config._attn_implementation = 'eager'
        o = rt.model(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True, output_attentions=True)
    finally:
        for m in rt.model.modules():
            if hasattr(m, 'config') and hasattr(m.config, '_attn_implementation'):
                m.config._attn_implementation = 'sdpa'
    bt = torch.as_tensor(box_tokens, device=rt.device) + s0
    share, intgt = [], []
    for a in o.attentions:                                   # (1, H, n_q, T)
        w = a[0].float()
        rows = torch.as_tensor([s_ - min(steps) for s_ in steps], device=w.device)
        w = w[:, rows]                                       # (H, n_steps, T)
        share.append(w[..., s0:s1].sum(-1).cpu())
        intgt.append(w[..., bt].sum(-1).cpu())
    del cache
    return {'share': torch.stack(share).numpy().round(4).tolist(), 'in_target': torch.stack(intgt).numpy().round(4).tolist()}


def vl_pairs(args) -> list[dict]:
    a = argparse.Namespace(work=args.work, data_dir=args.data_dir, n_refcoco=1600)
    cases = cmr.Cases(a)
    return cmr.grid_pairs(args.n_vlpairs) + cmr.refcoco_pairs(cases, args.n_vlpairs, 'point')


def box_tokens(truth: dict, W: int) -> list[int]:
    b = truth['box']
    i0, i1 = int(b[1] // 32), int((b[3] - 1e-6) // 32)
    j0, j1 = int(b[0] // 32), int((b[2] - 1e-6) // 32)
    return [i * W + j for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def stage_vlflow(rt, args) -> None:
    if rt.family != 'qwen3_vl':
        raise ValueError('vlflow needs Qwen3-VL')
    path = args.output / 'vlflow.jsonl'
    done = {r['id'] for r in read_jsonl(path)}
    heads_path = args.output / 'vl_heads.json'
    prs = vl_pairs(args)
    a_ns = argparse.Namespace(work=args.work)
    t0 = time.time()
    # pass 1: attention census on the first pairs of each family to fix the head set
    if not heads_path.exists():
        acc = {}
        for fam in ('grid10', 'refcoco_point'):
            for pr in [p for p in prs if p['family'] == fam][:args.n_census]:
                a, b, ta, tb = cmr.pair_images(pr, a_ns)
                prompt = point_prompt(pr['target'])
                gen = cmr.gen_batch(rt, [(a, prompt)], 48)[0]['ids']
                seq = cmr.Seq(rt, [a, b], prompt, gen)
                vals = cmr.studied_values(rt, pr, gen)
                if not vals:
                    continue
                W = a.width // 32
                steps = [seq.P - 1 + v['start'] for v in vals]
                try:
                    att = vl_attention(seq, steps, box_tokens(ta, W))
                except Exception as exc:                              # eager attention unavailable
                    print('attention census failed:', repr(exc), flush=True)
                    continue
                m = np.array(att['in_target']).mean(2)                # (L, H), mean over value steps
                acc.setdefault(fam, []).append(m)
        top = {}
        for fam, ms in acc.items():
            m = np.mean(ms, 0)
            idx = np.dstack(np.unravel_index(np.argsort(-m.ravel()), m.shape))[0][:6]
            top[fam] = [[int(l) + 1, int(h), float(m[l, h])] for l, h in idx]
        heads = []
        for fam in top:
            for l, h, _ in top[fam]:
                if (l, h) not in heads:
                    heads.append((l, h))
        for lh in ((25, 13), (24, 30)):                              # 3.14 r2 / r3 readers
            if lh not in heads:
                heads.append(lh)
        heads_path.write_text(json.dumps({'top': top, 'heads': heads}))
        print('heads', heads, flush=True)
    heads = [tuple(h) for h in json.loads(heads_path.read_text())['heads']]
    step_c, win_c = vl_conditions(heads)
    for j, pr in enumerate(prs):
        if pr['id'] in done:
            continue
        a, b, ta, tb = cmr.pair_images(pr, a_ns)
        prompt = point_prompt(pr['target'])
        gen = cmr.gen_batch(rt, [(a, prompt)], 48)[0]['ids']
        seq = cmr.Seq(rt, [a, b], prompt, gen)
        vals = cmr.studied_values(rt, pr, gen)
        W = a.width // 32
        steps_all = []
        recs = []
        for v in vals:
            first = seq.P - 1 + v['start']
            steps_all += [first - 3, first - 2, first - 1, first, first + 1, first + 2]
            for i in range(3):
                tau = first + i
                rec = {'role': v['role'], 'digit': i, 'value': v['value']}
                rec['step'] = vl_run(seq, tau, tau, step_c, heads)
                rec['win'] = vl_run(seq, tau, first - 3, win_c, heads)
                rec['tail'] = vl_tail(seq, tau, first)
                recs.append(rec)
        try:
            att = vl_attention(seq, sorted(set(steps_all)), box_tokens(ta, W)) if vals else None
        except Exception as exc:
            print('attention record failed:', repr(exc), flush=True)
            att = None
        append_jsonl(path, [{**pr, 'truthA': ta, 'truthB': tb, 'heads': heads,
                             'answerA': rt.tokenizer.decode(gen, skip_special_tokens=True)[-300:],
                             'values': recs, 'att_steps': sorted(set(steps_all)), 'P': seq.P,
                             'value_starts': [seq.P - 1 + v['start'] for v in vals], 'att': att}])
        print(f'vlflow {j + 1}/{len(prs)} {pr["family"]} {len(recs)} steps {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- main

STAGES = {'capture': stage_capture, 'roll': stage_roll, 'steer': stage_steer, 'vlflow': stage_vlflow}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--n-jgrid', type=int, default=160)
    p.add_argument('--n-single', type=int, default=64)
    p.add_argument('--n-refcoco', type=int, default=96)
    p.add_argument('--n-roll', type=int, default=64)
    p.add_argument('--n-steer', type=int, default=48)
    p.add_argument('--n-vlpairs', type=int, default=32)
    p.add_argument('--n-census', type=int, default=8)
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    rt = qr.load(args.model)
    if rt.family == 'qwen3_5':
        cv_use_torch_deltanet()
    meta = {'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt), 'code': code_digest(),
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'torch_deltanet': rt.family == 'qwen3_5'}
    (args.output / f"run_{args.stages.replace(',', '_')}_{time.strftime('%m%d_%H%M')}.json").write_text(
        json.dumps(meta, indent=1))
    for s in args.stages.split(','):
        STAGES[s](rt, args)


def cv_use_torch_deltanet() -> None:
    """Use the pure-torch chunked DeltaNet (the hub kernel needs more shared memory than A40s provide, and the
    DeltaTap wrapper must see the call)."""
    from analysis_tools.position.causal_vs_linear_run import use_torch_deltanet
    use_torch_deltanet()


if __name__ == '__main__':
    main()
