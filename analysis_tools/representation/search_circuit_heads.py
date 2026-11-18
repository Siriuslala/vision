"""Experiment 2.7 round 3, circuit stage: heads and neurons of the odd-entity localisation circuit (single images).

For a fixed set of correctly answered odd images (target after raster cell 6) and their corrupted counterparts:

  capture   one forward per odd and corrupted image, recording at the target tokens the inputs of o_proj in full
            attention layers 4/8/12 (per-head outputs), of out_proj in DeltaNet layers 1-15 (per-head outputs), of
            down_proj in MLP 8-15 (neuron activations) and the layer-15 output; plus
              * the full-attention layer-8 QK account of the target queries (logits and masses on the own cell and on
                earlier cells, odd vs corrupted);
              * reader QK: the row query of every reader head and the key change of the target (odd - corrupted);
              * direct attribution of every MLP 8-15 neuron to the L16 / L20 reader logits of the target
                (activation change x gradient of the reader logit through the frozen-norm key path, which is exact
                for the direct path).
  necessity per-head interventions at the target tokens of the odd image: L4/L8/L12 head h may not attend to other
            cells (fa{l}h{h}_blk); L4/L8/L12 head output replaced by the corrupted value (fa{l}h{h}_pat); DeltaNet
            9-15 head h memory reset alpha=0 (dn{l}h{h}_rst; comparison-class only).
  selection top L8 heads, top DeltaNet heads (largest mean drop of reader attention on comparison items) and top-k
            neurons (largest mean attribution on comparison items), random neuron controls.
  second    neuron necessity (top-k activations replaced by corrupted values) and sufficiency in the corrupted image
            (top heads' / neurons' / all MLP 8-15 / layer-15 values copied from the odd image at the target tokens).

Readout: attention of the six reader heads from the row query on the target cell (mean of heads).
Outputs: circuit/records.npz (capture), circuit/rows.jsonl (interventions), circuit/selection.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.common import qwen_runtime as qr
from analysis_tools.representation import odd_entity as oe
from analysis_tools.representation import search_asymmetry_run as sr
from analysis_tools.representation import search_circuit_analysis as A
from analysis_tools.representation import search_circuit_run as R

FA = (4, 8, 12)
DN_MID = (9, 10, 11, 13, 14, 15)
DN_ALL = tuple(l for l in range(1, 16) if l % 4)
MLP = tuple(range(8, 16))
NEURON_K = (10, 50, 200)


def select_items(items: R.Items, output: Path, n_cmp: int, n_det: int, seed: int) -> list[dict]:
    beh = {r['id']: r for r in read_jsonl(output / 'results_behave.jsonl')}
    pool = defaultdict(list)
    for it in sorted(items.odd(['atlas', 'mag']), key=lambda x: x['id']):
        if A.tcell(it) < 6 or it['param'] == 3.0 or not A.lenient(it, beh[it['id']])[0]:
            continue
        pool[A.feature_class(it['pair_name'], it['variant'])].append(it)
    rng = np.random.default_rng(seed)
    out = []
    for cls, n in (('comparison', n_cmp), ('detector', n_det)):
        p = pool[cls]
        out += [p[i] for i in sorted(rng.choice(len(p), size=min(n, len(p)), replace=False))]
    return out


# ----------------------------------------------------------------------------- hooks

@contextmanager
def capture_target(rt, tpos: list[int], L: int, inputs_at=(8, 16, 20)):
    """Record, at the target positions, the inputs of FA o_proj, DeltaNet out_proj, MLP down_proj and the layer-15
    output; and full layer inputs + position embeddings at `inputs_at`."""
    st = {'o': {}, 'dn': {}, 'act': {}, 'res15': None, 'inp': {}, 'pe': {}}
    pos = torch.as_tensor(tpos)
    hs = []

    def pre(store, key):
        def f(m, a):
            if a[0].shape[1] == L:
                store[key] = a[0][0, pos.to(a[0].device)].detach().clone()
        return f
    for l in FA:
        hs.append(rt.layers[l - 1].self_attn.o_proj.register_forward_pre_hook(pre(st['o'], l)))
    for l in DN_ALL:
        hs.append(rt.layers[l - 1].linear_attn.out_proj.register_forward_pre_hook(pre(st['dn'], l)))
    for l in MLP:
        hs.append(rt.layers[l - 1].mlp.down_proj.register_forward_pre_hook(pre(st['act'], l)))

    def h15(m, a, o):
        h = sr._out(o)
        if h.shape[1] == L:
            st['res15'] = h[0, pos.to(h.device)].detach().clone()
    hs.append(rt.layers[14].register_forward_hook(h15))
    for l in inputs_at:
        def pl(m, a, kw, l=l):
            if a[0].shape[1] == L:
                st['inp'][l] = a[0][0].detach()
                st['pe'][l] = tuple(x[0] for x in kw['position_embeddings'])
        hs.append(rt.layers[l - 1].register_forward_pre_hook(pl, with_kwargs=True))
    try:
        yield st
    finally:
        for h in hs:
            h.remove()


@contextmanager
def apply_edits(rt, specs: list[dict], groups: list[list[int]], L: int):
    """specs: one dict per batch row with optional keys
         'blk': [(layer, head)]                     cross-cell block of that head for the row's target queries
         'o': [(layer, head, values (4, 256))]      replace FA head output (o_proj input slice) at target tokens
         'dno': [(layer, head, values (4, 128))]    replace DeltaNet head output (out_proj input slice)
         'act': [(layer, idx LongTensor, values (4, k))]  replace MLP neuron activations
         'mlp': [(layer, values (4, 12288))]        replace all neuron activations of an MLP
         'res15': values (4, 4096)                  replace the layer-15 output
         'rst': [(layer, head)]                     DeltaNet alpha = 0 at the target tokens for that head
       and the key 'tpos' (absolute target positions)."""
    from analysis_tools.position.rowcol_mechanism_run import override_gates_heads
    dev = rt.layers[0].mlp.down_proj.weight.device
    hs = []
    by = defaultdict(list)
    for b, sp in enumerate(specs):
        tp = torch.as_tensor(sp['tpos'], device=dev)
        for l, h, v in sp.get('o', []):
            by[('o', l)].append((b, tp, h, v))
        for l, h, v in sp.get('dno', []):
            by[('dno', l)].append((b, tp, h, v))
        for l, idx, v in sp.get('act', []):
            by[('act', l)].append((b, tp, idx, v))
        for l, v in sp.get('mlp', []):
            by[('act', l)].append((b, tp, None, v))
        if 'res15' in sp:
            by[('res15', 15)].append((b, tp, None, sp['res15']))

    def slicer(width):
        def f(m, a, lst):
            x = a[0]
            if x.shape[1] != L:
                return None
            x = x.clone()
            for b, tp, h, v in lst:
                if h is None:
                    x[b, tp] = v.to(x.device, x.dtype)
                elif isinstance(h, int):
                    x[b, tp, h * width:(h + 1) * width] = v.to(x.device, x.dtype)
                else:
                    sub = x[b, tp]
                    sub[:, h.to(x.device)] = v.to(x.device, x.dtype)
                    x[b, tp] = sub
            return (x,)
        return f
    for (kind, l), lst in by.items():
        if kind == 'o':
            f = slicer(rt.layers[l - 1].self_attn.head_dim)
            hs.append(rt.layers[l - 1].self_attn.o_proj.register_forward_pre_hook(lambda m, a, f=f, lst=lst: f(m, a, lst)))
        elif kind == 'dno':
            f = slicer(rt.layers[l - 1].linear_attn.head_v_dim)
            hs.append(rt.layers[l - 1].linear_attn.out_proj.register_forward_pre_hook(lambda m, a, f=f, lst=lst: f(m, a, lst)))
        elif kind == 'act':
            f = slicer(1)
            hs.append(rt.layers[l - 1].mlp.down_proj.register_forward_pre_hook(lambda m, a, f=f, lst=lst: f(m, a, lst)))
        else:
            def h15(m, a, o, lst=lst):
                h = sr._out(o)
                if h.shape[1] != L:
                    return o
                for b, tp, _, v in lst:
                    h[b, tp] = v.to(h.device, h.dtype)
                return o
            hs.append(rt.layers[14].register_forward_hook(h15))
    # per-head cross-cell block for target queries
    blk_layers = sorted({l for sp in specs for l, _ in sp.get('blk', [])})
    if blk_layers:
        causal = torch.ones(L, L, dtype=torch.bool, device=dev).tril()
        H = rt.layers[blk_layers[0] - 1].self_attn.config.num_attention_heads
        for l in blk_layers:
            mask = causal[None, None].repeat(len(specs), H, 1, 1)
            for b, sp in enumerate(specs):
                heads = [h for ll, h in sp.get('blk', []) if ll == l]
                if not heads:
                    continue
                tp = sp['tpos']
                own = set(tp)
                tcell_tokens = next(g for g in groups if set(g) == own)
                others = torch.as_tensor([t for g in groups if g is not tcell_tokens for t in g], device=dev)
                q = torch.as_tensor(tp, device=dev)
                for h in heads:
                    mask[b, h, q[:, None], others[None, :]] = False

            def pre(m, a, kw, mask=mask):
                if kw['hidden_states'].shape[1] != L:
                    return a, kw
                kw['attention_mask'] = mask
                return a, kw
            hs.append(rt.layers[l - 1].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    gates = [{'reset': {l: (np.asarray(sp['tpos']), [h]) for l, h in sp.get('rst', [])}} for sp in specs]
    # override_gates_heads expects one (positions, heads) per layer: merge heads per layer
    for g, sp in zip(gates, specs):
        merged = defaultdict(list)
        for l, h in sp.get('rst', []):
            merged[l].append(h)
        g['reset'] = {l: (np.asarray(sp['tpos']), hl) for l, hl in merged.items()}
    try:
        with override_gates_heads(rt, gates, L):
            yield
    finally:
        for h in hs:
            h.remove()


# ----------------------------------------------------------------------------- QK accounts

def rope_inv(x, cos, sin):
    from transformers.models.qwen3_5.modeling_qwen3_5 import rotate_half
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    return torch.cat([xr * cos - rotate_half(xr) * sin, xp], -1)


def reader_gradient(rt, layer: int, heads: list[int], x: torch.Tensor, cos, sin, qpos: int, tpos: list[int]):
    """d(sum of the heads' logits on token j)/d x_j at the target tokens with frozen RMSNorm scales.
    Returns (len(tpos), 4096) float32 and the queries (len(heads), 256)."""
    blk = rt.layers[layer - 1]
    att = blk.self_attn
    hd = att.head_dim
    q = R.query_vectors(rt, layer, x, cos, sin, [qpos])[0]                  # (H, hd)
    xi = x[tpos].float()
    r_in = torch.rsqrt(xi.pow(2).mean(-1, keepdim=True) + blk.input_layernorm.eps)
    w_in = 1.0 + blk.input_layernorm.weight.float()
    kraw = (blk.input_layernorm(x[tpos])).float() @ att.k_proj.weight.float().T
    kraw = kraw.view(len(tpos), -1, hd)
    r_k = torch.rsqrt(kraw.pow(2).mean(-1, keepdim=True) + att.k_norm.eps)  # (T, KV, 1)
    w_k = 1.0 + att.k_norm.weight.float()
    Wk = att.k_proj.weight.float().view(-1, hd, x.shape[-1])                 # (KV, hd, d)
    g = torch.zeros(len(tpos), x.shape[-1], device=x.device)
    for h in heads:
        kv = h // att.num_key_value_groups
        dk = rope_inv(q[h][None].expand(len(tpos), -1), cos[tpos].float(), sin[tpos].float())   # (T, hd)
        dk = dk * w_k * r_k[:, kv]
        g += torch.einsum('td,tdc->tc', dk, Wk[kv][None].expand(len(tpos), -1, -1)) * r_in * w_in * att.scaling
    return g, q[heads]


def fa8_account(rt, x, cos, sin, tpos, groups, tcell) -> np.ndarray:
    """Layer-8 heads, target queries: [own-cell mean logit, earlier-cell mean logit, own mass, earlier mass] (H, 4)."""
    att = rt.layers[7].self_attn
    q = R.query_vectors(rt, 8, x, cos, sin, tpos)                            # (T, H, hd)
    k = R.key_vectors(rt, 8, x, cos, sin).repeat_interleave(att.num_key_value_groups, dim=1)
    pr = R.attention_probs(rt, 8, x, cos, sin, tpos)                         # (H, T, L)
    own = groups[tcell]
    earlier = [t for g in groups[:tcell] for t in g]
    lo = torch.einsum('thd,shd->hts', q, k[own]) * att.scaling
    le = torch.einsum('thd,shd->hts', q, k[earlier]) * att.scaling
    return torch.stack([lo.mean((1, 2)), le.mean((1, 2)), pr[:, :, own].sum(-1).mean(1),
                        pr[:, :, earlier].sum(-1).mean(1)], -1).float().cpu().numpy()


# ----------------------------------------------------------------------------- main loop

def readout(rt, st2, b, groups, qpos, tcell):
    rd = R.reader_readout(rt, st2, b, groups, qpos)
    return float(rd[:, tcell].mean()), bool(np.argmax(rd.sum(0)) == tcell)


def run_rows(rt, base, answers, i_row, i_col, P, groups, tcell, specs, args):
    out = []
    for a in range(0, len(specs), args.batch):
        chunk = specs[a:a + args.batch]
        batch = R.repeat_batch(base, len(chunk))
        with apply_edits(rt, chunk, groups, P['L']), R.capture_layers(rt, (16, 20), 0, None) as st2:
            logits = rt.model(**batch, use_cache=False, logits_to_keep=len(answers[0]) + 1).logits
        sc_ = sr.score_rows(logits, answers * len(chunk), i_row, i_col, len(answers[0]), sr.digit_ids(rt, 6))
        for b in range(len(chunk)):
            m, top = readout(rt, st2, b, groups, P['q_row'], tcell)
            out.append({'reader': m, 'top1': top, 's': sc_[b]['s']})
    return out


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n-cmp', type=int, default=80)
    p.add_argument('--n-det', type=int, default=40)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    torch.set_num_threads(16)
    out_dir = args.output / 'circuit'
    out_dir.mkdir(exist_ok=True)
    items = R.Items(args.output)
    sel = select_items(items, args.output, args.n_cmp, args.n_det, args.seed)
    rt = qr.load(args.model)
    t0 = time.time()
    rec = defaultdict(list)
    cache = []
    rows_path = out_dir / 'rows.jsonl'
    rows_path.write_text('')
    for n, it in enumerate(sel):
        cls = A.feature_class(it['pair_name'], it['variant'])
        tcell = A.tcell(it)
        runs = {}
        for role, iid in (('odd', it['id']), ('corr', it['partner'])):
            base, answers, i_row, i_col = R.batch_for(rt, items, [iid], [tuple(it['target'])])
            P = R.positions(rt, base, len(answers[0]), i_row, i_col)
            groups = R.cell_tokens(it['meta'], P['s'])
            tpos = groups[tcell]
            with capture_target(rt, tpos, P['L']) as st:
                rt.model(**base, use_cache=False, logits_to_keep=1)
            runs[role] = (base, st)
        st1, st0 = runs['odd'][1], runs['corr'][1]
        # QK accounts
        rec['id'].append(it['id']); rec['cls'].append(cls)
        rec['fa8'].append(np.stack([fa8_account(rt, s['inp'][8], *s['pe'][8], tpos, groups, tcell) for s in (st1, st0)]))
        qs, dks, attr = [], [], []
        for layer in (16, 20):
            heads = [h for l, h in R.READERS if l == layer]
            cos, sin = st1['pe'][layer]
            g, q = reader_gradient(rt, layer, heads, st1['inp'][layer], cos, sin, P['q_row'], tpos)
            k1 = R.key_vectors(rt, layer, st1['inp'][layer], cos, sin)[tpos].mean(0)
            k0 = R.key_vectors(rt, layer, st0['inp'][layer], *st0['pe'][layer])[tpos].mean(0)
            kvg = rt.layers[layer - 1].self_attn.num_key_value_groups
            qs.append(q.cpu().numpy()); dks.append(torch.stack([(k1 - k0)[h // kvg] for h in heads]).cpu().numpy())
            sc = []
            for l in MLP:
                W = rt.layers[l - 1].mlp.down_proj.weight.float()            # (d, I)
                dact = (st1['act'][l] - st0['act'][l]).float()              # (T, I)
                sc.append((dact * (g @ W)).sum(0))
            attr.append(torch.stack(sc).half().cpu().numpy())               # (8, I)
        rec['q'].append(np.stack(qs)); rec['dk'].append(np.stack(dks)); rec['attr'].append(np.stack(attr))
        rec['dres16'].append((st1['inp'][16][tpos] - st0['inp'][16][tpos]).float().mean(0).half().cpu().numpy())
        # necessity: per-head interventions on the odd image
        specs, names = [{'tpos': tpos}], ['none']
        for l in FA:
            for h in range(16):
                hd = 256
                specs.append({'tpos': tpos, 'blk': [(l, h)]}); names.append(f'fa{l}h{h}_blk')
                specs.append({'tpos': tpos, 'o': [(l, h, st0['o'][l][:, h * hd:(h + 1) * hd])]}); names.append(f'fa{l}h{h}_pat')
        if cls == 'comparison':
            for l in DN_MID:
                for h in range(32):
                    specs.append({'tpos': tpos, 'rst': [(l, h)]}); names.append(f'dn{l}h{h}_rst')
        base1, answers, i_row, i_col = R.batch_for(rt, items, [it['id']], [tuple(it['target'])])
        res = run_rows(rt, base1, answers, i_row, i_col, P, groups, tcell, specs, args)
        R._append(rows_path, [{'id': it['id'], 'cls': cls, 'phase': 'necessity', 'cond': c, **r} for c, r in zip(names, res)])
        cache.append((it, cls, tcell, tpos, groups, P, {k: {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                                                         if isinstance(v, dict) else (v.cpu() if torch.is_tensor(v) else v)
                                                         for k, v in st1.items() if k in ('o', 'dn', 'act', 'res15')},
                      {k: v for k, v in st0.items() if k == 'act'}))
        print(f'[circuit] necessity {n + 1}/{len(sel)} {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(out_dir / 'records.npz', **{k: np.asarray(v) for k, v in rec.items()})
    # ---------------------------------------------------------------- selection (comparison items)
    rows = read_jsonl(rows_path)
    base_r = {r['id']: r['reader'] for r in rows if r['cond'] == 'none'}
    drop = defaultdict(list)
    for r in rows:
        if r['cls'] == 'comparison' and r['cond'] != 'none':
            drop[r['cond']].append(base_r[r['id']] - r['reader'])
    md = {c: float(np.mean(v)) for c, v in drop.items()}
    fa8 = sorted(range(16), key=lambda h: -max(md[f'fa8h{h}_blk'], md[f'fa8h{h}_pat']))[:3]
    dnh = sorted([(l, h) for l in DN_MID for h in range(32)], key=lambda lh: -md[f'dn{lh[0]}h{lh[1]}_rst'])[:5]
    attr = np.asarray(rec['attr'], dtype=np.float32)                        # (n, 2, 8, I)
    cmp_mask = np.asarray(rec['cls']) == 'comparison'
    score = attr[cmp_mask].sum(1).mean(0)                                    # (8, I)
    order = np.argsort(-score.ravel())
    rng = np.random.default_rng(args.seed)
    top = {k: [(MLP[i // score.shape[1]], int(i % score.shape[1])) for i in order[:k]] for k in NEURON_K}
    rand = {k: [(MLP[i // score.shape[1]], int(i % score.shape[1])) for i in rng.choice(score.size, size=k, replace=False)]
            for k in NEURON_K}
    selection = {'fa8_heads': fa8, 'dn_heads': dnh, 'neurons_top': top, 'neurons_rand': rand,
                 'mean_drop': md, 'items': [it['id'] for it in sel]}
    (out_dir / 'selection.json').write_text(json.dumps(selection, indent=1))
    # ---------------------------------------------------------------- second pass
    dev = rt.layers[0].mlp.down_proj.weight.device

    def by_layer(nlist):
        d = defaultdict(list)
        for l, i in nlist:
            d[l].append(i)
        return {l: torch.as_tensor(v) for l, v in d.items()}
    for n, (it, cls, tcell, tpos, groups, P, s1, s0) in enumerate(cache):
        odd_specs, odd_names, cor_specs, cor_names = [], [], [], []
        for k in NEURON_K:
            for tag, nl in (('top', top[k]), ('rand', rand[k])):
                bl = by_layer(nl)
                odd_specs.append({'tpos': tpos, 'act': [(l, idx, s0['act'][l][:, idx.to(s0['act'][l].device)]) for l, idx in bl.items()]})
                odd_names.append(f'nec_nrn_{tag}{k}')
                cor_specs.append({'tpos': tpos, 'act': [(l, idx, s1['act'][l][:, idx]) for l, idx in bl.items()]})
                cor_names.append(f'suf_nrn_{tag}{k}')
        cor_specs.append({'tpos': tpos, 'o': [(8, h, s1['o'][8][:, h * 256:(h + 1) * 256]) for h in fa8]}); cor_names.append('suf_fa8top')
        cor_specs.append({'tpos': tpos, 'dno': [(l, h, s1['dn'][l][:, h * 128:(h + 1) * 128]) for l, h in dnh]}); cor_names.append('suf_dntop')
        cor_specs.append({'tpos': tpos, 'o': [(8, h, s1['o'][8][:, h * 256:(h + 1) * 256]) for h in fa8],
                          'dno': [(l, h, s1['dn'][l][:, h * 128:(h + 1) * 128]) for l, h in dnh]}); cor_names.append('suf_heads')
        cor_specs.append({'tpos': tpos, 'mlp': [(l, s1['act'][l]) for l in MLP]}); cor_names.append('suf_mlp8_15')
        cor_specs.append({'tpos': tpos, 'res15': s1['res15']}); cor_names.append('suf_res15')
        cor_specs.append({'tpos': tpos}); cor_names.append('corr_none')
        odd_specs.append({'tpos': tpos}); odd_names.append('odd_none')
        odd_specs.append({'tpos': tpos, 'blk': [(8, h) for h in fa8]}); odd_names.append('nec_fa8top_blk')
        odd_specs.append({'tpos': tpos, 'rst': list(dnh)}); odd_names.append('nec_dntop_rst')
        for role, iid, specs, names in (('odd', it['id'], odd_specs, odd_names), ('corr', it['partner'], cor_specs, cor_names)):
            base, answers, i_row, i_col = R.batch_for(rt, items, [iid], [tuple(it['target'])])
            res = run_rows(rt, base, answers, i_row, i_col, P, groups, tcell, specs, args)
            R._append(rows_path, [{'id': it['id'], 'cls': cls, 'phase': 'second', 'image': role, 'cond': c, **r}
                                  for c, r in zip(names, res)])
        print(f'[circuit] second {n + 1}/{len(cache)} {time.time() - t0:.0f}s', flush=True)
    (out_dir / 'run.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()}, **qr.describe(rt),
                                                  'seconds': time.time() - t0, 'batch_size': args.batch}, indent=1))


if __name__ == '__main__':
    main()
