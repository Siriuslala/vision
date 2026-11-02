"""Experiment 4.5 analysis: per-task decomposition of the DeepStack effect on natural fine-grained tasks.

Reads runs/<model>/<condition>.scores.jsonl of ``deepstack_natural`` and, for TextVQA (validation), POPE and
RefCOCO, the runs of experiments 4.3/4.4 (``deepstack_todo_analysis.load_runs``). Writes tables.md,
summary.json and the figures of docs/analysis/deepstack/results_2026-10-21_natural.md.

Contrasts (paired on identical items, difference of accuracies, bootstrap 95% interval, McNemar p):
  total      Full - Off        everything the injection contributes
  content    Full - Blank      image content of the injection beyond the content-free position code
  hf         Full - Blur2      detail finer than about 2 pixels
  lf         Blur8 - Blank     coarse (low-frequency) content
  position   Blank - Mean      per-token variation of the content-free injection (position code)
  stats      Mean - Off        first-order statistics of the injection
  content_pos  Full - Meanpos    image content beyond a position-only injection with average natural content
  position_pos Meanpos - Mean    position code on top of the average injection
  uniform      Meanpos - Blank   cost of the uniform-colour content that Blank adds (content conflict)
  region     Full - Blank-in   content at the evidence tokens only
  control    Full - Blank-ctrl content at an equally shaped non-evidence region
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from analysis_tools.behavior.decomposition import read_jsonl
from analysis_tools.deepstack.deepstack_comparison import slug
from analysis_tools.deepstack.deepstack_comparison_analysis import fmt_p, mcnemar, paired_bootstrap, rate
from analysis_tools.deepstack.deepstack_natural import REGION_TASKS, TASKS, TOKEN_PIXELS

CONTRASTS = (('total', 'full', 'off'), ('content', 'full', 'blank'), ('hf', 'full', 'blur2'),
             ('lf', 'blur8', 'blank'), ('position', 'blank', 'mean'), ('stats', 'mean', 'off'),
             ('content_pos', 'full', 'meanpos'), ('position_pos', 'meanpos', 'mean'), ('uniform', 'meanpos', 'blank'))
REGION_CONTRASTS = (('region', 'full', 'blank_in'), ('control', 'full', 'blank_ctrl'),
                    ('content', 'full', 'blank'))
CONTRAST_LABELS = {'total': 'Full − Off', 'content': 'Full − Blank', 'hf': 'Full − Blur2', 'lf': 'Blur8 − Blank',
                   'position': 'Blank − Mean', 'stats': 'Mean − Off', 'content_pos': 'Full − Meanpos',
                   'position_pos': 'Meanpos − Mean', 'uniform': 'Meanpos − Blank', 'region': 'Full − Blank-in',
                   'control': 'Full − Blank-ctrl', 'outside': 'Full − Blank-out'}
TASK_LABELS = {'textvqa': 'TextVQA small text', 'docvqa': 'DocVQA', 'gqa': 'GQA small object', 'cub': 'CUB attribute',
               'mmvp': 'MMVP', 'textvqa_val': 'TextVQA val (4.4)', 'pope': 'POPE (4.4)', 'refcoco': 'RefCOCO (4.3/4.4)'}
ORDER = ('full', 'blur2', 'blur8', 'meanpos', 'blank', 'mean', 'off', 'blankflip', 'blank_in', 'blank_ctrl', 'blank_out')
LABELS = {'meanpos': 'Meanpos', 'full': 'Full', 'blur2': 'Blur σ=2', 'blur8': 'Blur σ=8', 'blank': 'Blank', 'mean': 'Mean', 'off': 'Off',
          'blankflip': 'Blank-flip', 'blank_in': 'Blank-in', 'blank_ctrl': 'Blank-ctrl', 'blank_out': 'Blank-out'}
# Same condition colours as experiment 4.4 (results_2026-10-20_todo.md) so figures read across documents.
COLORS = {'meanpos': '#0f8f6a', 'full': '#52514e', 'blur2': '#f2a27c', 'blur8': '#b8441a', 'blank': '#6250d6', 'mean': '#1baf7a',
          'off': '#2a78d6', 'blankflip': '#c2408a', 'blank_in': '#e0a100', 'blank_ctrl': '#9c9a92',
          'blank_out': '#3d8f8f'}
CONTRAST_COLORS = {'content_pos': '#3a2a99', 'position_pos': '#0f8f6a', 'uniform': '#c2408a', 'total': '#2a78d6', 'position': '#1baf7a', 'content': '#6250d6', 'hf': '#f2a27c',
                   'lf': '#b8441a', 'stats': '#9c9a92', 'region': '#e0a100', 'control': '#9c9a92',
                   'outside': '#3d8f8f'}
MODELS = ('Qwen3-VL-8B-Instruct', 'Qwen3-VL-4B-Instruct')


# ----------------------------------------------------------------------------- loading


def load_natural(output: Path, model: str) -> dict[str, dict[str, dict]]:
    cases = {c['id']: c for c in read_jsonl(output / 'cases.jsonl')}
    runs: dict = {}
    for path in sorted((output / 'runs' / slug(model)).glob('*.scores.jsonl')):
        runs[path.name[:-len('.scores.jsonl')]] = {r['id']: {**cases[r['id']], **r} for r in read_jsonl(path)}
    return runs


def load_previous(todo: Path, model: str) -> dict[str, dict[str, dict]]:
    """TextVQA validation, POPE and RefCOCO rows of experiments 4.3/4.4 under the natural-task task names."""
    from analysis_tools.deepstack.deepstack_todo_analysis import load_runs
    ablation = todo.parent / '2026-10-15_ablation_light'
    old = load_runs(todo, ablation, model)
    names = {'TextVQA': 'textvqa_val', 'POPE': 'pope', 'RefCOCO': 'refcoco'}
    out: dict = {}
    for cond, rows in old.items():
        for cid, r in rows.items():
            if r['task'] in names:
                out.setdefault(cond, {})[cid] = {**r, 'task': names[r['task']]}
    return out


def merge(*parts: dict) -> dict:
    out: dict = {}
    for part in parts:
        for cond, rows in part.items():
            out.setdefault(cond, {}).update(rows)
    return out


# ----------------------------------------------------------------------------- statistics


def task_ids(runs: dict, task: str, conds: tuple[str, ...], need=lambda r: True) -> list[str]:
    """Items of ``task`` answered under every listed condition (paired comparisons never mix item sets)."""
    if any(c not in runs for c in conds):
        return []
    ids = [i for i, r in runs[conds[0]].items() if r['task'] == task and need(r)]
    return sorted(i for i in ids if all(i in runs[c] for c in conds))


def contrast(runs: dict, task: str, a: str, b: str, n_boot: int, seed: int, need=lambda r: True) -> dict | None:
    ids = task_ids(runs, task, (a, b), need)
    if not ids:
        return None
    x = np.array([bool(runs[a][i]['correct']) for i in ids], float)
    y = np.array([bool(runs[b][i]['correct']) for i in ids], float)
    t = mcnemar(list(x.astype(bool)), list(y.astype(bool)))
    out = {'n': len(ids), 'a': float(x.mean()), 'b': float(y.mean()), 'delta': float(x.mean() - y.mean()),
           'ci': paired_bootstrap(x, y, n_boot, seed), 'p': t['p']}
    cont = continuous(runs, ids, a, b, n_boot, seed)
    if cont is not None:
        out['cont'] = cont
    return out


def margin(r: dict) -> float | None:
    """Continuous evidence for the correct answer: gold log-probability, or the truth-signed yes/no margin."""
    if r['task'] in ('textvqa', 'docvqa', 'gqa'):
        return r.get('gold_logp')
    if r['task'] == 'cub':
        return r['s'] if r['answers'][0] == 'Yes' else -r['s']
    if r['task'] == 'pope':
        return r['s'] if r['truth']['answer_yes'] else -r['s']
    return None


def continuous(runs: dict, ids: list[str], a: str, b: str, n_boot: int, seed: int) -> dict | None:
    pairs = [(margin(runs[a][i]), margin(runs[b][i])) for i in ids]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 10:
        return None
    x, y = np.array(pairs, float).T
    return {'n': len(pairs), 'delta': float(np.mean(x - y)), 'ci': paired_bootstrap(x, y, n_boot, seed),
            'n_missing': len(ids) - len(pairs)}


def accuracy_table(runs: dict, tasks: tuple[str, ...]) -> dict:
    out: dict = {}
    for task in tasks:
        for cond in ORDER:
            rows = [r for r in runs.get(cond, {}).values() if r['task'] == task]
            if rows:
                out.setdefault(task, {})[cond] = rate([bool(r['correct']) for r in rows])
    return out


def mmvp_pairs(runs: dict, cond: str) -> dict | None:
    rows = [r for r in runs.get(cond, {}).values() if r['task'] == 'mmvp']
    if not rows:
        return None
    pairs: dict = {}
    for r in rows:
        pairs.setdefault(r['pair'], []).append(bool(r['correct']))
    full = [all(v) for v in pairs.values() if len(v) == 2]
    return rate(full)


# ----------------------------------------------------------------------------- figures


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def plot_decomposition(summary: dict, tasks: list[str], out: Path, model: str) -> None:
    plt = _plt()
    names = ('total', 'position_pos', 'content_pos', 'content', 'hf')
    tasks = [t for t in tasks if summary['contrasts'][model].get(t)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.16
    for k, name in enumerate(names):
        xs, ys, lo, hi = [], [], [], []
        for t, task in enumerate(tasks):
            c = summary['contrasts'][model].get(task, {}).get(name)
            if c is None:
                continue
            xs.append(t + (k - 2) * width)
            ys.append(c['delta'])
            lo.append(c['delta'] - c['ci'][0])
            hi.append(c['ci'][1] - c['delta'])
        ax.bar(xs, ys, width * 0.92, color=CONTRAST_COLORS[name], label=CONTRAST_LABELS[name],
               yerr=[lo, hi], capsize=2, error_kw={'lw': 0.8})
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xticks(range(len(tasks)), [TASK_LABELS[t] for t in tasks], rotation=15)
    ax.set_ylabel('accuracy difference (paired)')
    ax.set_title(f'{model}: decomposition of the DeepStack effect per task')
    ax.legend(ncol=5, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis='y', color='#e4e3df', lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_accuracy(summary: dict, tasks: list[str], out: Path, model: str) -> None:
    plt = _plt()
    tasks = [t for t in tasks if summary['accuracy'][model].get(t)]
    conds = [c for c in ORDER if any(c in summary['accuracy'][model].get(t, {}) for t in tasks)]
    mat = np.full((len(tasks), len(conds)), np.nan)
    for i, t in enumerate(tasks):
        for j, c in enumerate(conds):
            v = summary['accuracy'][model].get(t, {}).get(c)
            if v is not None:
                mat[i, j] = v['acc']
    full = mat[:, [conds.index('full')]]
    fig, ax = plt.subplots(figsize=(1.0 * len(conds) + 2.5, 0.55 * len(tasks) + 1.5))
    im = ax.imshow(mat - full, cmap='RdBu', vmin=-0.4, vmax=0.4, aspect='auto')
    for i in range(len(tasks)):
        for j in range(len(conds)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_xticks(range(len(conds)), [LABELS[c] for c in conds], rotation=30, ha='right')
    ax.set_yticks(range(len(tasks)), [TASK_LABELS[t] for t in tasks])
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('accuracy − Full accuracy')
    ax.set_title(f'{model}: accuracy per condition (cell text = accuracy)')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_region(summary: dict, out: Path, model: str) -> None:
    plt = _plt()
    names = ('region', 'control', 'content')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for panel, (ax, key, ylabel) in enumerate(zip(axes, ('acc', 'cont'),
                                                  ('accuracy difference', 'margin difference (log-prob or logit)'))):
        for k, name in enumerate(names):
            xs, ys, err = [], [], [[], []]
            for t, task in enumerate(REGION_TASKS):
                c = summary['region'][model].get(task, {}).get(name)
                if c is None:
                    continue
                v = c if key == 'acc' else c.get('cont')
                if v is None:
                    continue
                xs.append(t + (k - 1.5) * 0.2)
                ys.append(v['delta'])
                err[0].append(v['delta'] - v['ci'][0])
                err[1].append(v['ci'][1] - v['delta'])
            ax.bar(xs, ys, 0.18, color=CONTRAST_COLORS[name], label=CONTRAST_LABELS[name], yerr=err, capsize=2,
                   error_kw={'lw': 0.8})
        ax.axhline(0, color='#52514e', lw=0.8)
        ax.set_xticks(range(len(REGION_TASKS)), [TASK_LABELS[t] for t in REGION_TASKS], rotation=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#e4e3df', lw=0.6)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title('answer accuracy')
    axes[1].set_title('evidence for the correct answer')
    fig.suptitle(f'{model}: content-free replacement restricted to the evidence region (items with a control box)')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_models(summary: dict, tasks: list[str], out: Path) -> None:
    plt = _plt()
    series = [('Qwen3-VL-8B-Instruct', 'full', 'Qwen3-VL-8B Full', '#52514e'),
              ('Qwen3-VL-8B-Instruct', 'off', 'Qwen3-VL-8B Off', '#2a78d6'),
              ('Qwen3-VL-8B-Instruct', 'blank', 'Qwen3-VL-8B Blank', '#6250d6'),
              ('Qwen3-VL-4B-Instruct', 'full', 'Qwen3-VL-4B Full', '#9c9a92'),
              ('Qwen3-VL-4B-Instruct', 'off', 'Qwen3-VL-4B Off', '#8fb8ea'),
              ('Qwen3.5-9B', 'full', 'Qwen3.5-9B', '#e0a100')]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.13
    for k, (model, cond, label, color) in enumerate(series):
        xs, ys, err = [], [], [[], []]
        for t, task in enumerate(tasks):
            v = summary['accuracy'].get(model, {}).get(task, {}).get(cond)
            if v is None:
                continue
            xs.append(t + (k - 2.5) * w)
            ys.append(v['acc'])
            err[0].append(v['acc'] - v['ci'][0])
            err[1].append(v['ci'][1] - v['acc'])
        ax.bar(xs, ys, w * 0.92, color=color, label=label, yerr=err, capsize=1.5, error_kw={'lw': 0.7})
    ax.set_xticks(range(len(tasks)), [TASK_LABELS[t] for t in tasks])
    ax.set_ylabel('accuracy (Wilson 95%)')
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(axis='y', color='#e4e3df', lw=0.6)
    ax.set_title('Model comparison on the natural fine-grained tasks')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_examples(output: Path, runs: dict, out: Path, model: str, n: int = 4) -> None:
    """DocVQA items answered correctly under Full and wrongly under Blank: crop around the evidence box."""
    plt = _plt()
    raw = {r['id']: r for r in read_jsonl(output / 'runs' / slug(model) / 'blank.jsonl')}
    raw_full = {r['id']: r for r in read_jsonl(output / 'runs' / slug(model) / 'full.jsonl')}
    raw_off = {r['id']: r for r in read_jsonl(output / 'runs' / slug(model) / 'off.jsonl')}
    ids = [i for i, r in sorted(runs['blank'].items()) if r['task'] == 'docvqa' and not r['correct']
           and runs['full'][i]['correct'] and i in raw_off][:n]
    fig, axes = plt.subplots(1, len(ids), figsize=(4.4 * len(ids), 3.6))
    for ax, i in zip(np.atleast_1d(axes), ids):
        c = runs['full'][i]
        im = Image.open(output / c['image']).convert('RGB')
        x0 = min(b[0] for b in c['boxes']); y0 = min(b[1] for b in c['boxes'])
        x1 = max(b[2] for b in c['boxes']); y1 = max(b[3] for b in c['boxes'])
        pad = 48
        box = (max(0, int(x0) - 3 * pad), max(0, int(y0) - pad), min(im.width, int(x1) + 3 * pad),
               min(im.height, int(y1) + pad))
        crop = im.crop(box)
        d = ImageDraw.Draw(crop)
        d.rectangle([x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1]], outline=(224, 161, 0), width=2)
        ax.imshow(crop)
        ax.axis('off')
        q = c['question'] if len(c['question']) < 50 else c['question'][:47] + '...'
        ax.set_title(f"Q: {q}\nFull: {raw_full[i]['response'][:28]}\nOff: {raw_off[i]['response'][:28]}\n"
                     f"Blank: {raw[i]['response'][:28]}", fontsize=8, loc='left')
    fig.suptitle(f'{model}: DocVQA answers under Full, Off and Blank (yellow box = evidence region)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_inputs(output: Path, out: Path) -> None:
    """One example per region task: evidence tokens (yellow) and control tokens (grey) on the token grid."""
    plt = _plt()
    cases = read_jsonl(output / 'cases.jsonl')
    picks = []
    for task in REGION_TASKS:
        cands = [c for c in cases if c['task'] == task and c.get('control')]
        picks.append(cands[len(cands) // 3])
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.4))
    for ax, c in zip(axes, picks):
        im = Image.open(output / c['image']).convert('RGB')
        over = Image.new('RGBA', im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        cols = c['width'] // TOKEN_PIXELS
        for key, colour in (('region', (224, 161, 0, 110)), ('control', (120, 120, 120, 110))):
            for k in c[key] or []:
                r, q = divmod(k, cols)
                d.rectangle([q * 32, r * 32, q * 32 + 31, r * 32 + 31], fill=colour, outline=colour[:3] + (255,))
        ax.imshow(Image.alpha_composite(im.convert('RGBA'), over))
        q = c['question'] if len(c['question']) < 60 else c['question'][:57] + '...'
        ax.set_title(f"{TASK_LABELS[c['task']]}\nQ: {q}\nA: {c['answers'][0]}", fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def analyze(output: Path, todo: Path, figures: Path, n_boot: int, seed: int) -> None:
    out_dir = output / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    all_tasks = (*TASKS, 'textvqa_val', 'pope', 'refcoco')
    summary: dict = {'accuracy': {}, 'contrasts': {}, 'region': {}, 'mmvp_pairs': {}}
    for model in (*MODELS, 'Qwen3.5-9B'):
        runs = load_natural(output, model)
        if model != 'Qwen3.5-9B':
            runs = merge(load_previous(todo, model), runs)
        if not runs:
            continue
        summary['accuracy'][model] = accuracy_table(runs, all_tasks)
        summary['mmvp_pairs'][model] = {c: mmvp_pairs(runs, c) for c in ORDER if mmvp_pairs(runs, c)}
        if model == 'Qwen3.5-9B':
            continue
        summary['contrasts'][model] = {
            t: {name: v for name, a, b in CONTRASTS if (v := contrast(runs, t, a, b, n_boot, seed)) is not None}
            for t in all_tasks}
        summary['contrasts'][model]['flip'] = {
            t: v for t in all_tasks if (v := contrast(runs, t, 'blankflip', 'blank', n_boot, seed)) is not None}
        has_ctrl = lambda r: r.get('control') is not None
        summary['region'][model] = {
            t: {name: v for name, a, b in REGION_CONTRASTS
                if (v := contrast(runs, t, a, b, n_boot, seed, has_ctrl)) is not None}
            for t in REGION_TASKS}
        summary['region'][model]['all_items'] = {
            t: v for t in REGION_TASKS if (v := contrast(runs, t, 'full', 'blank_in', n_boot, seed)) is not None}
        # Region size: does the evidence-content effect grow when the evidence covers more tokens?
        size: dict = {}
        for t in ('textvqa', 'docvqa', 'gqa'):
            for lo, hi in ((1, 4), (5, 12), (13, 10 ** 6)):
                v = contrast(runs, t, 'full', 'blank_in', n_boot, seed,
                             lambda r, lo=lo, hi=hi: lo <= len(r.get('region') or []) <= hi)
                if v is not None:
                    size.setdefault(t, {})[f'{lo}-{hi if hi < 10 ** 6 else "inf"}'] = v
        summary['region'][model]['by_size'] = size
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    write_tables(summary, out_dir / 'tables.md')
    shown = [t for t in ('textvqa', 'docvqa', 'gqa', 'cub', 'mmvp', 'textvqa_val', 'pope', 'refcoco')]
    for model in MODELS:
        if model not in summary['contrasts']:
            continue
        tag = slug(model)
        plot_decomposition(summary, shown, figures / f'decomposition_{tag}.png', model)
        plot_accuracy(summary, shown, figures / f'accuracy_{tag}.png', model)
        if any(summary['region'][model].get(t) for t in REGION_TASKS):
            plot_region(summary, figures / f'region_{tag}.png', model)
        runs = load_natural(output, model)
        if all(c in runs for c in ('full', 'off', 'blank')):
            plot_examples(output, runs, figures / f'doc_examples_{tag}.png', model)
    plot_models(summary, list(TASKS), figures / 'models.png')
    plot_inputs(output, figures / 'inputs.png')
    print('wrote', out_dir / 'tables.md')


def _fmt(c: dict | None) -> str:
    if c is None:
        return '–'
    s = f"{c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] (p={fmt_p(c['p'])})"
    return s


def _fmt_cont(c: dict | None) -> str:
    if c is None or 'cont' not in c:
        return '–'
    v = c['cont']
    return f"{v['delta']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"


def write_tables(summary: dict, path: Path) -> None:
    lines = ['# Experiment 4.5 tables', '']
    for model, acc in summary['accuracy'].items():
        conds = [c for c in ORDER if any(c in v for v in acc.values())]
        lines += [f'## {model}: accuracy', '', '| task | n | ' + ' | '.join(LABELS[c] for c in conds) + ' |',
                  '|---|---|' + '---|' * len(conds)]
        for task, row in acc.items():
            n = row.get('full', next(iter(row.values())))['n']
            lines.append(f'| {TASK_LABELS[task]} | {n} | ' + ' | '.join(
                f"{row[c]['acc']:.3f}" if c in row else '–' for c in conds) + ' |')
        if summary['mmvp_pairs'].get(model):
            lines += ['', 'MMVP pair accuracy: ' + ', '.join(
                f"{LABELS[c]} {v['acc']:.3f} (n={v['n']})" for c, v in summary['mmvp_pairs'][model].items())]
        lines.append('')
    for model, per_task in summary['contrasts'].items():
        lines += [f'## {model}: contrasts (accuracy difference [95% CI] (McNemar p))', '',
                  '| task | ' + ' | '.join(CONTRAST_LABELS[n] for n, _, _ in CONTRASTS) + ' | Blank-flip − Blank |',
                  '|---|' + '---|' * (len(CONTRASTS) + 1)]
        for task, row in per_task.items():
            if task == 'flip' or not row:
                continue
            lines.append(f'| {TASK_LABELS[task]} | ' + ' | '.join(_fmt(row.get(n)) for n, _, _ in CONTRASTS)
                         + f" | {_fmt(per_task['flip'].get(task))} |")
        lines += ['', f'### {model}: margin contrasts (gold log-prob or truth-signed yes/no logit margin)', '',
                  '| task | ' + ' | '.join(CONTRAST_LABELS[n] for n, _, _ in CONTRASTS) + ' |',
                  '|---|' + '---|' * len(CONTRASTS)]
        for task, row in per_task.items():
            if task == 'flip' or not any('cont' in v for v in row.values()):
                continue
            lines.append(f'| {TASK_LABELS[task]} | ' + ' | '.join(_fmt_cont(row.get(n)) for n, _, _ in CONTRASTS) + ' |')
        lines.append('')
    for model, per_task in summary['region'].items():
        lines += [f'## {model}: evidence-region replacement (items with a control box)', '',
                  '| task | n | ' + ' | '.join(CONTRAST_LABELS[n] for n, _, _ in REGION_CONTRASTS) + ' | margin: '
                  + ' | margin: '.join(CONTRAST_LABELS[n] for n, _, _ in REGION_CONTRASTS) + ' |',
                  '|---|---|' + '---|' * (2 * len(REGION_CONTRASTS))]
        for task in REGION_TASKS:
            row = per_task.get(task, {})
            if not row:
                continue
            n = next(iter(row.values()))['n']
            lines.append(f'| {TASK_LABELS[task]} | {n} | ' + ' | '.join(_fmt(row.get(k)) for k, _, _ in REGION_CONTRASTS)
                         + ' | ' + ' | '.join(_fmt_cont(row.get(k)) for k, _, _ in REGION_CONTRASTS) + ' |')
        lines += ['', 'Full − Blank-in on all items: ' + '; '.join(
            f"{TASK_LABELS[t]} n={v['n']} {_fmt(v)}" for t, v in per_task.get('all_items', {}).items()), '',
            'Full − Blank-in by number of evidence tokens: ' + '; '.join(
                f"{TASK_LABELS[t]} {k}: n={v['n']} {_fmt(v)} margin {_fmt_cont(v)}"
                for t, d in per_task.get('by_size', {}).items() for k, v in d.items()), '']
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--todo', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    analyze(a.output, a.todo, a.figures, a.n_boot, a.seed)


if __name__ == '__main__':
    main()
