"""Experiment 3.6 round 2 stage drivers (GPU). Design: design_coordinate_value.md, section "第二轮".

Stages (comma list via --stages):
  formats        greedy answers and a teacher-forced pass on the TRUE answer for numeric-format variants
                 of the disc point task, dot counting, RefCOCO point questions and a blank-canvas prior;
                 records the digit distribution of every digit given the true preceding digits and the
                 residuals at the digit-predicting positions.
  neurons        MLP neuron activations (down_proj input) of layers 20-32 at the roles x0, x1, y0, y1 on
                 all round-1 disc samples (teacher-forced on the model's own answers).
  ablate         mean ablation of single components at the first-digit position (x0 or y0) and greedy
                 continuation of the number.
  pathpatch      L20 H11 activation patching from a donor image (sender) and mediation through the MLP,
                 token-mixer and head outputs of later layers at the first-digit position.
  neuron_ablate  mean ablation of selected MLP neurons (from the CPU analysis) at the first-digit position.
Round-1 records, cases and features are read from --round1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position import coordinate_value as cv
from analysis_tools.position import coordinate_value_r2 as c2
from analysis_tools.position.coordinate_readout_run import append_jsonl, done_ids, read_jsonl

RESID_LAYERS = (16, 20, 24, 28, 32)
NEURON_LAYERS = tuple(range(20, 33))
NEURON_ROLES = ('x0', 'x1', 'y0', 'y1')
DISC_FORMATS = ('s1000', 's100', 's10', 'pix', 'frac', 'text')
COUNT_RANGE = range(1, 51)
COUNT_SEEDS = 5
ABLATE_MLP = tuple(range(18, 33))
PAIR_STEPS = 38                     # donor offset along the sweep (2-pixel steps, about 148 units)
N_PAIRS = 60                        # pairs per axis
MED_MLP = tuple(range(20, 33))
MED_MIXER = tuple(range(21, 33))
MED_HEAD_LAYERS = (24, 28, 32)
SENDER = (20, 11)


def code_digest() -> dict:
    root = Path(__file__).parent
    names = ('coordinate_value_r2.py', 'coordinate_value_r2_run.py', 'coordinate_value.py', 'coordinate_readout.py')
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()[:16] for n in names}


def round1(args):
    r1 = Path(args.round1)
    recs = read_jsonl(r1 / 'gen' / 'records.jsonl')
    cases = {c['id']: c for c in json.loads((r1 / 'cases.json').read_text())}
    return {r['id']: r for r in recs}, cases


def sweep_ids(recs: dict, name: str, every: int = 1) -> list[str]:
    ids = sorted([i for i, r in recs.items() if r['set'] == name], key=lambda i: recs[i]['truth_units'][name == 'sweep_v'])
    return ids[::every]


# ----------------------------------------------------------------------------- formats

def format_cases(args) -> list[dict]:
    recs, cases = round1(args)
    out = []
    for fmt in DISC_FORMATS:
        for name in ('sweep_h', 'sweep_v'):
            for i in sweep_ids(recs, name, 2):
                c = cases[i]
                t = recs[i]['truth_units']
                if fmt == 'text':
                    px, py = (int(round(v)) for v in c['center'])
                    prompt = c2.TEXT_QUERY.format(px=px, py=py)
                    t = [1000 * px / c2.CANVAS, 1000 * py / c2.CANVAS]
                else:
                    prompt = c2.format_prompt(fmt)
                out.append({'id': f'{fmt}_{i}', 'kind': 'disc', 'fmt': fmt, 'source': i, 'sweep': name,
                            'center': c['center'], 'prompt': prompt, 'truth': t,
                            'true_strings': [c2.true_string(fmt, v) for v in t]})
    rc = json.loads((Path(args.refcoco_cases)).read_text())
    for c in rc:
        if c['format'] != 'point':
            continue
        W, H = c['truth']['canvas']
        t = [1000 * c['truth']['center'][0] / W, 1000 * c['truth']['center'][1] / H]
        for fmt in ('s1000', 's100'):
            prompt = c['prompt'].replace(cr.COORD_NOTE, c2.FORMAT_NOTES[fmt])
            if fmt == 's1000' and prompt != c['prompt']:
                raise ValueError('RefCOCO prompt does not end with the 0-1000 note')
            out.append({'id': f"rc_{fmt}_{c['id']}", 'kind': 'refcoco', 'fmt': fmt, 'source': c['id'],
                        'image_path': c['image_path'], 'image_sha256': c['image_sha256'], 'prompt': prompt,
                        'truth': t, 'true_strings': [c2.true_string(fmt, v) for v in t]})
    for n in COUNT_RANGE:
        for s in range(COUNT_SEEDS):
            out.append({'id': f'count_{n:02d}_{s}', 'kind': 'count', 'fmt': 'count', 'n': n, 'seed': 1000 * n + s,
                        'prompt': c2.COUNT_QUERY, 'truth': [float(n)], 'true_strings': [str(n)]})
    for fmt, values in (('s1000', [f'{d}55' for d in range(1, 10)]), ('s100', [f'{d}5' for d in range(1, 10)])):
        for v in values:
            out.append({'id': f'blank_{fmt}_{v}', 'kind': 'blank', 'fmt': fmt, 'prompt': c2.format_prompt(fmt),
                        'truth': [float('nan')] * 2, 'true_strings': [v, v]})
    return out


def format_input(case: dict):
    if case['kind'] == 'disc':
        img, _ = cr.disc_image(tuple(case['center']), cv.DIAMETER, c2.CANVAS)
    elif case['kind'] == 'refcoco':
        p = Path(case['image_path'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != case['image_sha256']:
            raise ValueError(f'Image hash mismatch for {p}')
        img = Image.open(p).convert('RGB')
    elif case['kind'] == 'count':
        img, _ = c2.count_image(case['n'], case['seed'])
    else:
        img = Image.new('RGB', (c2.CANVAS, c2.CANVAS), 'white')
    return img, case['prompt']


def stage_formats(rt, out: Path, args) -> None:
    root = out / 'formats'
    root.mkdir(parents=True, exist_ok=True)
    cpath = root / 'cases.json'
    if not cpath.exists():
        cpath.write_text(json.dumps(format_cases(args), indent=1))
    cases = json.loads(cpath.read_text())
    gpath = root / 'gen.jsonl'
    finished = done_ids(gpath)
    # Greedy answers, batched over equal-length prompts (no padding).
    t0 = time.time()
    preps = {}
    groups: dict = {}
    todo = [c for c in cases if c['id'] not in finished]
    for c in todo:
        img, text = format_input(c)
        p = cr.prepare(rt, img, text)
        preps[c['id']] = p
        groups.setdefault(len(p.ids), []).append(c)
        if len(preps) >= 4 * args.batch_size:
            _flush_gen(rt, groups, preps, gpath, args, force=False)
    _flush_gen(rt, groups, preps, gpath, args, force=True)
    print(f'formats gen done {time.time() - t0:.0f}s', flush=True)
    # Teacher-forced pass on the true answer.
    gen = {r['id']: r for r in read_jsonl(gpath)}
    tpath = root / 'tf.jsonl'
    finished = done_ids(tpath)
    todo = [c for c in cases if c['id'] not in finished]
    digit_ids = cv.digit_token_ids(rt)
    shard_i = len(list(root.glob('tf_shard_*.npz')))
    for s in range(0, len(todo), args.batch_size):
        chunk = todo[s:s + args.batch_size]
        xs, ps, sel, metas = [], [], [], []
        for c in chunk:
            img, text = format_input(c)
            prep = cr.prepare(rt, img, text)
            if c['kind'] == 'count':
                ans, spans, templ = c['true_strings'][0], [(0, len(c['true_strings'][0]))], True
            else:
                ans, templ = c2.ideal_point_text(gen[c['id']]['text'], c['true_strings'])
                spans = c2.point_numbers(ans)
            ids, slots = c2.digit_slots(rt.tokenizer, len(prep.ids), ans, spans)
            x, p = cr.teacher_forced(rt, prep, ids)
            xs.append(x)
            ps.append(p)
            sel.append([sl.pos for sl in slots])
            metas.append((c, ans, templ, slots, ids))
        X, P = c2.pad_batch(xs, ps)
        res = c2.select_forward(rt, X, P, sel, RESID_LAYERS)
        rows, resid = [], []
        for (c, ans, templ, slots, ids), r in zip(metas, res):
            lp = r['logp']
            dl = lp[:, digit_ids]
            pd = dl.exp().sum(-1)                                              # total probability of a digit token
            rows.append({'id': c['id'], 'answer': ans, 'model_template': templ, 'shard': shard_i,
                         'slots': [sl.__dict__ for sl in slots], 'digit_logp': dl.tolist(),
                         'p_digit': pd.tolist(), 'top_token': [rt.tokenizer.decode([int(t)]) for t in lp.argmax(-1)],
                         'row_offset': sum(len(x['slots']) for x in rows)})
            resid.append(np.stack([r['resid'][l].numpy() for l in RESID_LAYERS], 1))   # (S, Lr, D)
        np.savez(root / f'tf_shard_{shard_i:04d}.npz', resid=np.concatenate(resid), ids=np.array([m[0]['id'] for m in metas]))
        append_jsonl(tpath, rows)
        shard_i += 1
        if (s // args.batch_size) % 10 == 0:
            print(f'formats tf {s + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)


def _flush_gen(rt, groups, preps, gpath, args, force: bool) -> None:
    for L in list(groups):
        g = groups[L]
        while len(g) >= args.batch_size or (force and g):
            chunk, g[:] = g[:args.batch_size], g[args.batch_size:]
            res = cr.run_conditions(rt, [cr.embeds(rt, preps[c['id']]) for c in chunk], [preps[c['id']].pos for c in chunk],
                                    args.max_new_tokens, batch_size=args.batch_size)
            append_jsonl(gpath, [{'id': c['id'], 'text': r['text'], 'ids': r['ids'], 'finish_reason': r['finish_reason'],
                                  'prompt_len': len(preps[c['id']].ids), 'batch_size': len(chunk)} for c, r in zip(chunk, res)])
            for c in chunk:
                preps.pop(c['id'])
        if not g:
            groups.pop(L)


# ----------------------------------------------------------------------------- neurons

def stage_neurons(rt, out: Path, args) -> None:
    root = Path(args.features) / 'neurons'
    root.mkdir(parents=True, exist_ok=True)
    recs, cases = round1(args)
    ids = [i for i, r in recs.items() if r['set'] != 't6' and r['spans']]
    t0 = time.time()
    for s in range(0, len(ids), args.batch_size):
        shard = root / f'shard_{s // args.batch_size:04d}.npz'
        if shard.exists():
            continue
        chunk = ids[s:s + args.batch_size]
        xs, ps, sel = [], [], []
        for i in chunk:
            img, _, text = cv.disc_input(cases[i])
            prep = cr.prepare(rt, img, text)
            roles = cv.query_positions(len(prep.ids), recs[i]['spans'])
            x, p = cr.teacher_forced(rt, prep, recs[i]['ids'])
            xs.append(x)
            ps.append(p)
            sel.append([roles.get(r, roles['x0']) for r in NEURON_ROLES])       # 2-digit values reuse x0 (masked later)
        X, P = c2.pad_batch(xs, ps)
        res = c2.select_forward(rt, X, P, sel, (), NEURON_LAYERS)
        acts = np.stack([np.stack([r['neurons'][l].numpy() for l in NEURON_LAYERS], 1) for r in res])  # (B, 4, 13, F)
        np.savez(shard, ids=np.array(chunk), acts=acts)
        print(f'neurons {s + len(chunk)}/{len(ids)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- ablation helpers

def component_means(args) -> dict:
    """Dataset means at x0 / y0 of MLP and mixer outputs and of full-attention head slices (round-1 features)."""
    feat = Path(args.round1) / 'features'
    index = json.loads((feat / 'index.json').read_text())
    s_mlp = s_attn = s_head = None
    n = 0
    for sh in index['shards']:
        z = np.load(feat / sh['file'])
        keep = np.array([not i.startswith('t6') for i in z['ids']])
        m, a, h = (z[k][keep].astype(np.float64).sum(0) for k in ('mlp', 'attn', 'heads'))
        s_mlp, s_attn, s_head = (m, a, h) if s_mlp is None else (s_mlp + m, s_attn + a, s_head + h)
        n += int(keep.sum())
    return {'mlp': torch.tensor(s_mlp / n, dtype=torch.float32), 'mixer': torch.tensor(s_attn / n, dtype=torch.float32),
            'heads': torch.tensor(s_head / n, dtype=torch.float32), 'n': n}


def ablation_conditions() -> dict[str, list[tuple[str, int, int | None]]]:
    """name -> list of (site, layer, head) components ablated together."""
    conds: dict = {'clean': []}
    for l in ABLATE_MLP:
        conds[f'mlp{l}'] = [('mlp', l, None)]
    for name, ls in (('mlp28-30', (28, 29, 30)), ('mlp24-27', (24, 25, 26, 27)), ('mlp31-32', (31, 32)),
                     ('mlp20-32', tuple(range(20, 33)))):
        conds[name] = [('mlp', l, None) for l in ls]
    for l in (20, 24, 28, 32):
        conds[f'attn{l}'] = [('mixer', l, None)]
    conds['lin25-31'] = [('mixer', l, None) for l in (25, 26, 27, 29, 30, 31)]
    for l, h in ((32, 10), (32, 8), (24, 5), (24, 6), (20, 11)):
        conds[f'head{l}_{h}'] = [('head', l, h)]
    conds['zero_mlp28-30'] = [('mlp', l, 'zero') for l in (28, 29, 30)]
    return conds


def base_rows(args, every: int = 2) -> list[tuple[str, int]]:
    recs, _ = round1(args)
    return [(i, 0) for i in sweep_ids(recs, 'sweep_h', every)] + [(i, 1) for i in sweep_ids(recs, 'sweep_v', every)]


def prefix_input(rt, recs, cases, i: str, axis: int, image_case: str | None = None):
    """Inputs of prompt (image of image_case, default i) + the answer prefix of i up to the first digit of axis."""
    r = recs[i]
    img, _, text = cv.disc_input(cases[image_case or i])
    prep = cr.prepare(rt, img, text)
    prefix = r['ids'][:r['spans'][axis][0]]
    x, p = cv.prefix_inputs(rt, prep, prefix)
    return x, p, len(prep.ids)


def make_ops(row: int, comps, pos: int, means: dict, axis: int) -> list[c2.Op]:
    ops = []
    for site, l, h in comps:
        if site == 'mlp':
            v = torch.zeros(4096) if h == 'zero' else means['mlp'][l - 1, axis]
            ops.append(c2.Op(row, l, 'mlp', pos, v))
        elif site == 'mixer':
            ops.append(c2.Op(row, l, 'mixer', pos, means['mixer'][l - 1, axis]))
        elif site == 'head':
            ops.append(c2.Op(row, l, 'head', pos, means['heads'][cv.FULL_LAYERS.index(l), axis, h], h))
    return ops


def stage_ablate(rt, out: Path, args, conds: dict | None = None, path_name: str = 'ablate', neuron_means=None) -> None:
    root = out / path_name
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    recs, cases = round1(args)
    means = component_means(args) if neuron_means is None else None
    conds = conds or ablation_conditions()
    names = list(conds)
    t0 = time.time()
    for n, (i, axis) in enumerate(base_rows(args)):
        rid = f'{i}_{"xy"[axis]}'
        if rid in finished:
            continue
        x, p, _ = prefix_input(rt, recs, cases, i, axis)
        pos = x.shape[1] - 1
        rows = []
        for s in range(0, len(names), args.max_rows):
            chunk = names[s:s + args.max_rows]
            ops = []
            for b, name in enumerate(chunk):
                if neuron_means is None:
                    ops += make_ops(b, conds[name], pos, means, axis)
                else:
                    for l, idx in conds[name]:
                        ops.append(c2.Op(b, l, 'neuron', pos, neuron_means[(l, axis)][idx], torch.tensor(idx)))
            with c2.Interventions(rt, ops):
                res = cv.continue_number(rt, x.expand(len(chunk), -1, -1).contiguous(), p.expand(-1, len(chunk), -1).contiguous())
            for name, r in zip(chunk, res):
                rows.append({'cond': name, 'digits': r['digits'], 'stop': r['stop'], 'first_logp': r['first_logp']})
        append_jsonl(path, [{'id': rid, 'case': i, 'axis': axis, 'truth': recs[i]['truth_units'][axis],
                             'emitted_round1': recs[i]['score']['coords'][axis], 'conditions': rows,
                             'max_rows': args.max_rows}])
        if n % 20 == 0:
            print(f'{path_name} {n} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- path patching

def pair_list(args) -> list[tuple[str, str, int]]:
    recs, _ = round1(args)
    pairs = []
    for name, axis in (('sweep_h', 0), ('sweep_v', 1)):
        ids = sweep_ids(recs, name)
        for k in np.linspace(0, len(ids) - 1, N_PAIRS).round().astype(int):
            j = k + PAIR_STEPS if k < len(ids) // 2 else k - PAIR_STEPS
            pairs.append((ids[k], ids[j], axis))
    return pairs


def mediation_rows() -> list[tuple[str, str, object]]:
    """(name, kind, spec) rows of the second forward; every row uses run A's inputs."""
    rows = [(f'med_mlp{l}', 'set', [('mlp', l, None)]) for l in MED_MLP]
    rows += [('med_mlp24-31', 'set', [('mlp', l, None) for l in range(24, 32)]),
             ('med_mlp28-30', 'set', [('mlp', l, None) for l in (28, 29, 30)]),
             ('med_mlp20-32', 'set', [('mlp', l, None) for l in MED_MLP])]
    rows += [(f'med_mixer{l}', 'set', [('mixer', l, None)]) for l in MED_MIXER]
    rows += [(f'med_head{l}_{h}', 'set', [('head', l, h)]) for l in MED_HEAD_LAYERS for h in range(16)]
    rows += [('h11_x0_only', 'copy', 'x0'), ('h11_before_x0', 'copy', 'before'),
             ('P_block_mlp24-31', 'block', list(range(24, 32))), ('P_block_mlp28-30', 'block', [28, 29, 30]),
             ('P_block_mixer21-32', 'block_mixer', list(MED_MIXER))]
    return rows


def stage_pathpatch(rt, out: Path, args) -> None:
    root = out / 'pathpatch'
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'records.jsonl'
    finished = done_ids(path)
    recs, cases = round1(args)
    digit_ids = cv.digit_token_ids(rt)
    L20, H11 = SENDER
    med = mediation_rows()
    t0 = time.time()
    for n, (a, b, axis) in enumerate(pair_list(args)):
        rid = f'{a}__{b}'
        if rid in finished:
            continue
        xa, pa, plen = prefix_input(rt, recs, cases, a, axis)
        xb, pb, _ = prefix_input(rt, recs, cases, a, axis, image_case=b)
        L = xa.shape[1]
        x0, start = L - 1, plen - 1
        rec = c2.Record([x0] * 3, mlp=tuple(range(18, 33)), mixer=tuple(range(18, 33)), heads=(20,) + MED_HEAD_LAYERS)
        X = torch.cat([xa, xb, xa])
        P = torch.cat([pa, pb, pa], 1)
        ops = [c2.Op(2, L20, 'head', (start, L), 1, H11)]
        lp1 = c2.digit_logits_at(rt, X, P, [x0] * 3, digit_ids, ops, rec)
        D = rec.data
        rows = {'A': lp1[0].tolist(), 'B': lp1[1].tolist(), 'P': lp1[2].tolist()}
        # Second forward(s): row 0 is run B (copy source), the others are run A with edits.
        for s in range(0, len(med), args.max_rows - 1):
            chunk = med[s:s + args.max_rows - 1]
            ops = []
            for r, (name, kind, spec) in enumerate(chunk, start=1):
                if kind == 'set':
                    for site, l, h in spec:
                        if site == 'mlp':
                            ops.append(c2.Op(r, l, 'mlp', x0, D[('mlp', l)][2]))
                        elif site == 'mixer':
                            ops.append(c2.Op(r, l, 'mixer', x0, D[('mixer', l)][2]))
                        else:
                            ops.append(c2.Op(r, l, 'head', x0, D[('head', l)][2, h], h))
                elif kind == 'copy':
                    span = (x0, L) if spec == 'x0' else (start, x0)
                    ops.append(c2.Op(r, L20, 'head', span, 0, H11))
                elif kind in ('block', 'block_mixer'):
                    ops.append(c2.Op(r, L20, 'head', (start, L), 0, H11))
                    site = 'mlp' if kind == 'block' else 'mixer'
                    for l in spec:
                        ops.append(c2.Op(r, l, site, x0, D[(site, l)][0]))
            Xm = torch.cat([xb] + [xa] * len(chunk))
            Pm = torch.cat([pb] + [pa] * len(chunk), 1)
            lp = c2.digit_logits_at(rt, Xm, Pm, [x0] * (len(chunk) + 1), digit_ids, ops)
            for (name, _, _), v in zip(chunk, lp[1:]):
                rows[name] = v.tolist()
        append_jsonl(path, [{'id': rid, 'a': a, 'b': b, 'axis': axis, 'truth_a': recs[a]['truth_units'][axis],
                             'truth_b': recs[b]['truth_units'][axis], 'logp': rows,
                             'sender_delta_x0': (D[('head', L20)][2, H11] - D[('head', L20)][0, H11]).norm().item()}])
        if n % 10 == 0:
            print(f'pathpatch {n} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------- neuron ablation

def stage_neuron_ablate(rt, out: Path, args) -> None:
    sel = json.loads((out / 'neuron_sets.json').read_text())
    z = np.load(out / 'neuron_means.npz')
    neuron_means = {(int(l), a): torch.tensor(z[f'L{l}_{"xy"[a]}']) for l in sel['layers'] for a in (0, 1)}
    conds = {'clean': []}
    for name, items in sel['sets'].items():
        by: dict = {}
        for l, i in items:
            by.setdefault(int(l), []).append(int(i))
        conds[name] = list(by.items())
    stage_ablate(rt, out, args, conds, 'neuron_ablate', neuron_means)


def stage_spotcheck(rt, out: Path, args, n: int = 20) -> None:
    """Answer-level agreement of the batched manual greedy decoder (formats stage) with the reference generate()."""
    root = out / 'formats'
    cases = [c for c in json.loads((root / 'cases.json').read_text()) if c['kind'] != 'blank']
    gen = {r['id']: r for r in read_jsonl(root / 'gen.jsonl')}
    path = root / 'spotcheck.jsonl'
    finished = done_ids(path)
    rng = np.random.default_rng(20261016)
    for j in sorted(rng.choice(len(cases), size=n, replace=False)):
        c = cases[j]
        if c['id'] in finished:
            continue
        img, text = format_input(c)
        ref = qr.generate(rt, qr.build(rt, [img], text), args.max_new_tokens)
        a, b = ref['text'].strip(), gen[c['id']]['text'].strip()
        import re
        nums = lambda t: re.findall(r'\d+(?:\.\d+)?', t.split('"label"')[0])
        append_jsonl(path, [{'id': c['id'], 'reference': a, 'manual': b, 'text_equal': a == b,
                             'answer_equal': nums(a) == nums(b)}])


STAGES = {'spotcheck': stage_spotcheck, 'formats': stage_formats, 'neurons': stage_neurons, 'ablate': stage_ablate, 'pathpatch': stage_pathpatch,
          'neuron_ablate': stage_neuron_ablate}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--round1', required=True)
    p.add_argument('--refcoco-cases', required=True)
    p.add_argument('--stages', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-rows', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=48)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rt = qr.load(args.model)
    meta = out / 'experiment.json'
    info = json.loads(meta.read_text()) if meta.exists() else {'runs': []}
    info['runs'].append({'time': datetime.now(timezone.utc).isoformat(), 'stages': args.stages, 'runtime': qr.describe(rt),
                         'code': code_digest(), 'batch_size': args.batch_size, 'max_rows': args.max_rows,
                         'max_new_tokens': args.max_new_tokens})
    meta.write_text(json.dumps(info, indent=1))
    for s in args.stages.split(','):
        t0 = time.time()
        STAGES[s](rt, out, args)
        print(f'stage {s} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
