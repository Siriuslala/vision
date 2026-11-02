"""Experiment 3.9 round 3, GPU stage: flash-attention-compatible inputs at the slot.

  capture_heads  one prefill of prompt + call prefix (identical to the round-2 capture) per case of a
                 stratified subset; at the slot (last prefix token) stores the o_proj input (gated head
                 outputs, 16 heads x 256) of layers 16, 20 and 24, and for every visual token the
                 pre-RoPE key and the value of the layer-20 KV head shared by H11 (the KV cache content
                 a deployed tool would read).  No attention probabilities are computed.
  fit_vec        CPU readers on these inputs: ridge on head-output vectors (H11 only, layer 20, layers
                 16/20/24; direct and residual on the rough point) and a learned query over the
                 layer-20 key/value cache of the visual tokens.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from analysis_tools.position import readout_tool as rt_
from analysis_tools.position import readout_tool_r2 as r2
from analysis_tools.position import readout_tool_r2_analysis as a2
from analysis_tools.position import readout_tool_r2_run as r2run
from analysis_tools.position import readout_tool_run as r1run
from analysis_tools.position.coordinate_readout_run import read_jsonl

OPROJ_LAYERS = (16, 20, 24)
KV_LAYER = 20
MAP_LAYERS = (20, 24)                # attention maps of all heads of these layers at the sites below
MAP_SITES = (-3, -2, -1)             # offsets from the end of the prefix; -1 is the slot
H11 = 11
SEED = 20261021
N_TRAIN = {'refcoco': 4000, 'synth': 2500, 'graded': 1500}
N_TEST_MAX = 150                     # per non-RefCOCO test set
N_TEST_REFCOCO = 600
N_VAL_MAX = 600
CHUNK = 512


def subset_cases(cases: list[dict]) -> list[dict]:
    """Stratified subset: training draws per group, N_VAL_MAX validation cases, N_TEST_REFCOCO RefCOCO
    test cases and up to N_TEST_MAX cases of every other test set (canonical phrasing only)."""
    rng = np.random.default_rng(SEED)
    out = []
    for g, n in N_TRAIN.items():
        pool = [c for c in cases if c['split'] == 'train' and a2.val_group(c['set']) == g]
        out += [pool[i] for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False))]
    pool = [c for c in cases if c['split'] == 'val']
    out += [pool[i] for i in sorted(rng.choice(len(pool), min(N_VAL_MAX, len(pool)), replace=False))]
    for s in sorted({c['set'] for c in cases}):
        pool = [c for c in cases if c['split'] == 'test' and c['phrasing'] != 'heldout' and c['set'] == s]
        n = N_TEST_REFCOCO if s == 'refcoco' else N_TEST_MAX
        if len(pool) > n:
            pool = [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]
        out += pool
    return out


@torch.inference_mode()
def capture_heads(rt, texts: list[str], images) -> list[dict]:
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    vis = batch['input_ids'] == rt.model.config.image_token_id
    store, handles = {}, []
    for layer in OPROJ_LAYERS:
        def pre(module, args, layer=layer):
            store[f'o{layer}'] = args[0][:, -1].half().cpu()               # left padding: slot is the last position
        handles.append(rt.layers[layer - 1].self_attn.o_proj.register_forward_pre_hook(pre))
    mod = rt.layers[KV_LAYER - 1].self_attn
    kv_head = H11 // mod.num_key_value_groups

    def kv(module, args, kwargs):
        h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
        B, L, _ = h.shape
        k = module.k_norm(module.k_proj(h).view(B, L, -1, module.head_dim))[:, :, kv_head]
        v = module.v_proj(h).view(B, L, -1, module.head_dim)[:, :, kv_head]
        m = vis.to(h.device)
        store['k'] = [k[b][m[b]].half().cpu() for b in range(B)]
        store['v'] = [v[b][m[b]].half().cpu() for b in range(B)]
    handles.append(mod.register_forward_pre_hook(kv, with_kwargs=True))
    L = batch['input_ids'].shape[1]
    idx = torch.tensor([[L + o for o in MAP_SITES]] * len(texts), device=rt.device)
    centers = [rt_.token_centers(g) for g in batch['image_grid_thw'].tolist()]
    for layer in MAP_LAYERS:
        def mp(module, args, kwargs, layer=layer):
            h = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            d = h.device
            store[f'maps{layer}'] = [rt_._attention_features(module, h, kwargs['position_embeddings'], idx.to(d),
                                                             batch['attention_mask'].to(d), vis.to(d), centers, si)[1]
                                     for si in range(len(MAP_SITES))]
        handles.append(rt.layers[layer - 1].self_attn.register_forward_pre_hook(mp, with_kwargs=True))
    try:
        rt.model(**batch, use_cache=False, logits_to_keep=1)
    finally:
        for hd in handles:
            hd.remove()
    return [{'o': np.stack([store[f'o{l}'][b].numpy() for l in OPROJ_LAYERS]), 'k': store['k'][b].numpy(),
             'v': store['v'][b].numpy(),
             # (sites, layers * 16 heads, Nv)
             'maps': np.stack([np.concatenate([store[f'maps{l}'][si][b].numpy() for l in MAP_LAYERS], 0)
                               for si in range(len(MAP_SITES))])} for b in range(len(texts))]


def stage_capture_heads(args) -> None:
    from analysis_tools.common import qwen_runtime as qr
    cases = subset_cases(read_jsonl(args.r2 / 'cases.jsonl'))
    k_shard, n_shard = args.shard
    cases = [c for i, c in enumerate(cases) if i % n_shard == k_shard]
    feat = args.out / 'heads'
    feat.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in sorted(feat.glob(f'chunk_s{k_shard}_*.npz')):
        done |= set(np.load(f)['ids'].tolist())
    # evaluation cases and graded training scenes first, so that a run cut short by GPU availability
    # still leaves complete validation/test sets (fit_vec uses whatever training cases were captured)
    prio = lambda c: 0 if c['split'] != 'train' else (1 if c['set'].startswith('g_') else 2)
    todo = sorted((c for c in cases if c['id'] not in done), key=prio)
    print(f'shard {k_shard}/{n_shard}: {len(todo)} of {len(cases)} to do; groups {Counter(c["split"] for c in cases)}', flush=True)
    if not todo:
        return
    rt = rt_.load_runtime(args.model, [float(v) for v in args.max_memory.split(',')] if args.max_memory else None)
    (args.out / f'runtime_heads_s{k_shard}.json').write_text(json.dumps(qr.describe(rt), indent=1, default=str))
    ia = argparse.Namespace(r1_images=args.r1 / 'images', images=args.r2 / 'images', behavior_root=args.behavior_root)
    k0 = len(list(feat.glob(f'chunk_s{k_shard}_*.npz')))
    t0, n = time.time(), 0
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        rows, ids = [], []
        for b in r1run.batches(chunk, args.max_tokens, args.max_batch):
            texts = [r2.tool_prompt_r2(rt, r2run.user_text(c)) + r2.call_prefix(c['target']) for c in b]
            rows += capture_heads(rt, texts, [r2run.load_image(c, ia) for c in b])
            ids += [c['id'] for c in b]
            n += len(b)
        lens = np.asarray([len(r['k']) for r in rows])
        np.savez(feat / f'chunk_s{k_shard}_{k0:05d}.npz', ids=np.asarray(ids), o=np.stack([r['o'] for r in rows]),
                 n_vis=lens, k=np.concatenate([r['k'] for r in rows]), v=np.concatenate([r['v'] for r in rows]),
                 maps=np.concatenate([r['maps'] for r in rows], 2))
        k0 += 1
        print(f'captured {n}/{len(todo)}  {(time.time() - t0) / n:.3f} s/case', flush=True)
    (args.out / f'capture_heads_s{k_shard}.json').write_text(json.dumps(
        {'n_cases': len(cases), 'max_tokens': args.max_tokens, 'max_batch': args.max_batch, 'oproj_layers': OPROJ_LAYERS,
         'kv_layer': KV_LAYER, 'kv_head_of': H11, 'map_layers': MAP_LAYERS, 'map_sites': MAP_SITES, 'seconds': time.time() - t0}, indent=1))


# ----------------------------------------------------------------------------- 3.13 test set

def tf_calls(test_jsonl: Path) -> list[dict]:
    """Unique (image, target) calls of the experiment-3.13 test set; point and bbox items of the same
    image and target share one tool call (the tool returns both)."""
    calls = {}
    for r in read_jsonl(test_jsonl):
        key = f'{Path(r["image"]).stem}|{r["label"]}'
        calls.setdefault(key, {'id': key, 'image': r['image'], 'target': r['label'], 'family': r['family'],
                               'truth': r['truth'], 'items': []})['items'].append(r['id'])
    return list(calls.values())


def stage_capture_tf(args) -> None:
    """Round-2 capture protocol (canonical phrasing, model-filled rough point, slot attention of all
    128 heads, layer-0/16 content PCA) on the experiment-3.13 test images."""
    from PIL import Image
    from analysis_tools.common import qwen_runtime as qr
    calls = tf_calls(args.tf_test)
    feat = args.out / 'tf_features'
    feat.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in sorted(feat.glob('chunk_*.npz')):
        done |= set(np.load(f)['ids'].tolist())
    todo = [c for c in calls if c['id'] not in done]
    print(f'3.13 test: {len(todo)} of {len(calls)} calls to do', flush=True)
    if not todo:
        return
    rt = rt_.load_runtime(args.model, [float(v) for v in args.max_memory.split(',')] if args.max_memory else None)
    pca = torch.load(args.r2 / 'pca.pt')
    for c in todo:                                  # r1run.batches sorts by visual-token count
        W, H = c['truth']['canvas']
        c['n_visual'] = (W // 32) * (H // 32)
    k0 = len(list(feat.glob('chunk_*.npz')))
    t0, n = time.time(), 0
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        rows, ids = [], []
        for b in batches_by_tokens(chunk, args.max_tokens, args.max_batch):
            texts = [r2.tool_prompt_r2(rt, r2.PHRASINGS[0].format(target=c['target'])) + r2.call_prefix(c['target']) for c in b]
            offs = [r2.prefix_sites(rt.tokenizer, c['target']) for c in b]
            rows += r2.capture_r2(rt, texts, [Image.open(c['image']).convert('RGB') for c in b], offs, pca)
            ids += [c['id'] for c in b]
            n += len(b)
        r2run.save_chunk(feat / f'chunk_{k0:05d}.npz', ids, rows)
        k0 += 1
        print(f'captured {n}/{len(todo)}  {(time.time() - t0) / n:.3f} s/call', flush=True)
    (args.out / 'capture_tf.json').write_text(json.dumps({'n_calls': len(calls), 'max_tokens': args.max_tokens,
                                                          'max_batch': args.max_batch, 'seconds': time.time() - t0,
                                                          'runtime': qr.describe(rt)}, indent=1, default=str))


def batches_by_tokens(calls: list[dict], max_tokens: int, max_batch: int):
    calls = sorted(calls, key=lambda c: c['n_visual'])
    cur = []
    for c in calls:
        if cur and ((len(cur) + 1) * (c['n_visual'] + 200) > max_tokens or len(cur) == max_batch):
            yield cur
            cur = []
        cur.append(c)
    if cur:
        yield cur


# ----------------------------------------------------------------------------- CPU readers

def load_heads(out: Path) -> dict:
    ids, o, k, v, maps = [], [], [], [], []
    for f in sorted((out / 'heads').glob('chunk_s*_*.npz')):
        d = np.load(f)
        cut = np.cumsum(np.concatenate([[0], d['n_vis']]))
        ids += d['ids'].tolist()
        o.append(d['o'])
        kk, vv, mm = d['k'], d['v'], d['maps']                     # read each array once
        k += [kk[a:b] for a, b in zip(cut[:-1], cut[1:])]
        v += [vv[a:b] for a, b in zip(cut[:-1], cut[1:])]
        maps += [mm[:, :, a:b] for a, b in zip(cut[:-1], cut[1:])]
    return {'ids': ids, 'o': np.concatenate(o), 'k': k, 'v': v, 'maps': maps}


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Z = (X - mu) / sd
    ym = Y.mean(0)
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - ym))
    return mu, sd, W, ym


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    mu, sd, W, ym = model
    return ((X - mu) / sd) @ W + ym


def stage_fit_vec(args) -> None:
    cases = {c['id']: c for c in read_jsonl(args.r2 / 'cases.jsonl')}
    hd = load_heads(args.out)
    ids = hd['ids']
    split = np.asarray([cases[i]['split'] for i in ids])
    sets = np.asarray([cases[i]['set'] for i in ids])
    Y = np.stack([rt_.targets_units(cases[i]['truth']) for i in ids])
    # rough point and slot hidden state from the round-2 capture (same prefill, same slot)
    r2ids = json.loads((args.r2 / 'tool_r2' / 'ids.json').read_text())
    pos = {c: j for j, c in enumerate(r2ids)}
    cap_cont, cap_hidden = {}, {}
    for f in sorted((args.r2 / 'features').glob('chunk_s*_*.npz')):
        d = np.load(f)
        want = set(ids)
        cont, hid = d['cont'], d['hidden']
        for j, cid in enumerate(d['ids'].tolist()):
            if cid in want:
                cap_cont[cid] = str(cont[j])
                cap_hidden[cid] = hid[j, r2.HIDDEN_LAYERS.index(20)]
    missing = [i for i in ids if i not in cap_cont]
    if missing:
        raise ValueError(f'{len(missing)} cases missing from the round-2 capture')
    A = np.asarray([np.clip(r2.parse_approx(cap_cont[i]) or [500, 500], 0, 1000) for i in ids], float)
    o = hd['o'].astype(np.float32)                                         # (N, 3, 4096)
    li = OPROJ_LAYERS.index(20)
    inputs = {'o_L20H11': o[:, li, H11 * 256:(H11 + 1) * 256], 'o_L20': o[:, li], 'o_L16L20L24': o.reshape(len(o), -1)}
    tr, va = split == 'train', split == 'val'
    res, preds = {}, {}
    for name, X in inputs.items():
        for resid in (False, True):
            target = Y - np.concatenate([A, A, A], 1) if resid else Y
            best = None
            for lam in (1.0, 10.0, 100.0, 1000.0, 1e4):
                m = ridge_fit(X[tr], target[tr], lam)
                P = ridge_predict(m, X) + (np.concatenate([A, A, A], 1) if resid else 0)
                v = a2.val_metric(P[va], Y[va], sets[va])['mean']
                if best is None or v < best[0]:
                    best = (v, lam, P)
            key = f'ridge_{name}' + ('_resid' if resid else '')
            preds[key] = best[2]
            res[key] = {'lambda': best[1], 'val_mean': best[0], 'dim': int(X.shape[1])}
            print(key, res[key], flush=True)
    # learned query (slot hidden state, layer 20) over the layer-20 key/value cache of the visual tokens
    items = []
    for j, cid in enumerate(ids):
        g = rt_.grid_of(cases[cid]['truth']['canvas'])
        F = np.concatenate([hd['k'][j], hd['v'][j]], 1)
        if len(F) != g[0] * g[1]:
            raise ValueError(f'visual-token count mismatch for {cid}')
        items.append({'F': F.astype(np.float16), 'grid': g, 'q': rt_.token_overlap(Y[j, 2:], g), 'y': Y[j], 'id': cid,
                      'h': cap_hidden[cid].astype(np.float32)})
    S = {'train': tr, 'val': va}
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    params, hist = a2.fit_query_pointer(items, S, dev)
    P = a2.predict_query_pointer(params, items, dev)
    preds['query_kvcache_L20'] = P
    res['query_kvcache_L20'] = {'val_mean': a2.val_metric(P[va], Y[va], sets[va])['mean'], 'best_epoch': hist['best_epoch'],
                                'n_params': int(sum(v.numel() for v in params['state'].values()))}
    preds['approx_point'] = np.concatenate([A, A, A], 1)
    res['approx_point'] = {'val_mean': a2.val_metric(preds['approx_point'][va], Y[va], sets[va])['mean']}
    # attention at the slot and the two preceding tokens (zero-training peak decoder, per head)
    from analysis_tools.position import readout_tool_r3 as r3
    sites = {}
    te = split == 'test'
    for si, off in enumerate(MAP_SITES):
        for li, layer in enumerate(MAP_LAYERS):
            for h in range(16):
                P = np.stack([r3.peak_centroid(hd['maps'][j][si, li * 16 + h].astype(np.float32),
                                               rt_.grid_of(cases[ids[j]]['truth']['canvas'])) for j in range(len(ids))])
                sites[f'{off}|L{layer}H{h}'] = P
    site_res = {}
    for key, P in sites.items():
        site_res[key] = {}
        for s_ in sorted(set(sets[te])):
            m = te & (sets == s_)
            hits = [rt_.inside(rt_.to_pixels(P[j], cases[ids[j]]['truth']['canvas']), cases[ids[j]]['truth']['hit']) for j in np.nonzero(m)[0]]
            site_res[key][s_] = {'hit': float(np.mean(hits)), 'median': float(np.median(np.linalg.norm(P[m] - Y[m, :2], axis=1)))}
    (args.out / 'sites_vec.json').write_text(json.dumps(site_res))
    # per-set summary of all readers on the capture subset (test split), with the reduced CNN tools on the same cases
    r3preds = {}
    for f in sorted(args.out.glob('predictions_P*.npz')):
        r3preds.update(dict(np.load(f)))
    r2preds = {}
    for f in ('C', 'E'):
        r2preds.update(dict(np.load(args.r2 / 'tool_r2' / f'predictions_{f}.npz')))
    for name, P in (('cnn_k1', r3preds['k1_H11_content_approx']), ('cnn_all128', r2preds['cnn_slot_content_approx']),
                    ('cnn_k0', r2preds['cnn_content_approx'])):
        preds[name] = P[[pos[c] for c in ids]]
    summ = {}
    for name, P in preds.items():
        summ[name] = {}
        for s_ in sorted(set(sets[te])):
            m = te & (sets == s_)
            hits = [rt_.inside(rt_.to_pixels(P[j, :2], cases[ids[j]]['truth']['canvas']), cases[ids[j]]['truth']['hit']) for j in np.nonzero(m)[0]]
            summ[name][s_] = {'n': int(m.sum()), 'hit': float(np.mean(hits)), 'median': float(np.median(np.linalg.norm(P[m, :2] - Y[m, :2], axis=1)))}
    labels = {'ridge_o_L20H11': 'ridge on H11 output vector (256)', 'ridge_o_L20H11_resid': 'ridge on H11 output, residual to rough point',
              'ridge_o_L20': 'ridge on layer-20 head outputs (4096)', 'ridge_o_L20_resid': 'ridge on layer-20 outputs, residual',
              'ridge_o_L16L20L24': 'ridge on layer-16/20/24 outputs (12288)', 'ridge_o_L16L20L24_resid': 'ridge on layer-16/20/24 outputs, residual',
              'query_kvcache_L20': 'learned query over layer-20 K/V cache', 'approx_point': "model's rough point",
              'cnn_k1': 'one-head tool (attention row)', 'cnn_all128': '128-head tool', 'cnn_k0': 'content + rough point (no attention)'}
    (args.out / 'vec_summary.json').write_text(json.dumps({'readers': summ, 'labels': labels, 'fits': res}, indent=1))
    np.savez(args.out / 'predictions_vec.npz', ids=np.asarray(ids), **preds)
    (args.out / 'readers_vec.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--r2', type=Path, required=True)
    p.add_argument('--r1', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--max-memory', default='')
    p.add_argument('--max-tokens', type=int, default=8000)
    p.add_argument('--max-batch', type=int, default=32)
    p.add_argument('--behavior-root', type=Path, default=None)
    p.add_argument('--tf-test', type=Path, default=None, help='experiment-3.13 test.jsonl')
    args = p.parse_args()
    args.shard = tuple(int(v) for v in args.shard.split('/'))
    if args.behavior_root is None:
        args.behavior_root = args.r2.parents[1] / 'behavior'
    {'capture_heads': stage_capture_heads, 'fit_vec': stage_fit_vec, 'capture_tf': stage_capture_tf}[args.stage](args)


if __name__ == '__main__':
    main()
