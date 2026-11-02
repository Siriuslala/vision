"""Experiment 1.7 mechanism figures: channel-swap outcomes, image-attention timelines and token classes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from analysis_tools.behavior import cot_mechanism as cm
from analysis_tools.behavior import cot_mechanism_analysis as ca
from analysis_tools.behavior import cot_mechanism_engine as ce

COND_ORDER = ['clean_A', 'mask_img', 'img_from_B', 'txt_from_B', 'delta_from_B', 'txt_delta_from_B', 'all_from_B',
              'B_prefill_img_from_A', 'clean_B']
COND_LABEL = {
    'clean_A': 'clean A', 'mask_img': 'A;\nimage hidden\nin CoT', 'img_from_B': 'A;\nimage KV\nfrom B',
    'txt_from_B': 'A;\nquestion KV\nfrom B', 'delta_from_B': 'A;\nDeltaNet\nfrom B',
    'txt_delta_from_B': 'A;\nquestion KV +\nDeltaNet from B', 'all_from_B': 'A;\nall three\nfrom B',
    'B_prefill_img_from_A': 'B;\nimage KV\nfrom A', 'clean_B': 'clean B',
}


def fig_swap(mech: dict, path: Path, key: str = 'swap', title: str = 'final answer') -> None:
    tasks = [t for t in ('rowcol', 'odd', 'point') if any(k.startswith(t + '|') for k in mech[key])]
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10.5, 2.2 * len(tasks) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    cats = (['A', 'nearA', 'B', 'nearB', 'other', 'invalid'] if key == 'swap' else ['A', 'B', 'other', 'none'])
    colors = {'A': ca.PALETTE[0], 'nearA': '#86b6ef', 'B': ca.PALETTE[1], 'nearB': '#f4a582', 'other': ca.GREY,
              'invalid': '#d9d8d4', 'none': '#d9d8d4'}
    labels = {'A': 'equals A', 'nearA': 'closer to A', 'B': 'B', 'nearB': 'closer to B',
              'other': 'equidistant / other', 'invalid': 'no answer', 'none': 'no stated row'}
    labels['B'] = 'equals B'
    if key == 'swap':
        labels['A'] = 'equals A (point: nearer A)'
        labels['B'] = 'equals B (point: nearer B)'
    for ax, t in zip(axes, tasks):
        conds = [c for c in COND_ORDER if f'{t}|{c}' in mech[key]]
        for i, c in enumerate(conds):
            d = mech[key][f'{t}|{c}']
            left = 0
            for cat in cats:
                v = d[cat]['p'] if cat in d else 0
                if v > 0:
                    ax.bar(i, v, bottom=left, color=colors[cat], width=0.72, edgecolor='white', linewidth=1)
                    if v >= 0.12:
                        ax.text(i, left + v / 2, f'{100 * v:.0f}', ha='center', va='center', fontsize=7,
                                color='white' if cat in ('A', 'B', 'other') else ca.INK)
                left += v
        ax.set_xticks(range(len(conds)), [COND_LABEL[c] for c in conds], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f'{t}\nshare (n={mech[key][f"{t}|clean_A"]["n"]})', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in cats]
    axes[-1].legend(handles, [labels[c] for c in cats], ncol=len(cats), fontsize=7, frameon=False,
                    loc='upper center', bbox_to_anchor=(0.5, -0.42))
    axes[0].set_title(f'Decode-time channel swaps: which image does the {title} follow?', loc='left', fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def fig_attention_summary(att: dict, path: Path) -> None:
    tasks = [t for t in ('rowcol', 'odd', 'point') if t in att]
    L = list(ce.FULL_LAYERS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, t in enumerate(tasks):
        a = att[t]
        axes[0].plot(L, [a['step0_image_mass_by_layer'][str(l)] for l in L], '--', color=ca.PALETTE[i], lw=1.5)
        axes[0].plot(L, [a['mean_image_mass_by_layer'][str(l)] for l in L], '-o', color=ca.PALETTE[i], lw=2, ms=4, label=t)
        axes[1].plot(L, [a['target_share_by_layer'][str(l)] for l in L], '-o', color=ca.PALETTE[i], lw=2, ms=4, label=t)
        axes[1].axhline(a['target_share_uniform'], color=ca.PALETTE[i], ls=':', lw=1)
    axes[0].set_xlabel('full-attention layer')
    axes[0].set_ylabel('attention mass on image tokens')
    axes[0].set_title('(a) image mass per generated token\n(solid: mean over CoT; dashed: first step)', loc='left', fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel('full-attention layer')
    axes[1].set_ylabel('share of image attention on target cell')
    axes[1].set_title('(b) target-cell share (dotted: uniform)', loc='left', fontsize=8)
    # token classes at the layer with the largest mean mass
    t0 = tasks[0]
    best = max(L, key=lambda l: att[t0]['mean_image_mass_by_layer'][str(l)])
    classes = [c for c in ca.TOKEN_CLASSES if c in att[t0]['by_class']]
    w = 0.8 / len(tasks)
    for i, t in enumerate(tasks):
        vals = [att[t]['by_class'].get(c, {}).get(str(best), {}).get('mean', np.nan) for c in classes]
        axes[2].bar(np.arange(len(classes)) + (i - (len(tasks) - 1) / 2) * w, vals, width=w * 0.92, color=ca.PALETTE[i], label=t)
    axes[2].set_xticks(range(len(classes)), classes, rotation=30, ha='right', fontsize=7)
    axes[2].set_ylabel(f'image mass at layer {best}')
    axes[2].set_title(f'(c) by class of the produced token (layer {best})', loc='left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_timeline(run: Path, task: str, tokenizer, path: Path, row: int = 0, batch: int = 0) -> None:
    """One response: image mass per full-attention layer over the generated tokens, with token labels."""
    z = np.load(run / f'mech/probes/{task}_b{batch}_clean_A.npz')
    recs = [r for r in ca.read(run / 'mech/records.jsonl') if r['batch_id'] == f'{task}|b{batch}|clean_A']
    r = sorted(recs, key=lambda x: x['row_in_batch'])[row]
    n = r['n_tokens']
    m = z['img'][:n, row].astype(np.float32).mean(-1).T            # (L, n)
    toks = [tokenizer.decode([i]) for i in r['ids'][:n]]
    gh, gw = ca._grid_tokens(r)
    tmask = ca._cell_mask(r['truth_a'], gh, gw)
    maps = z['map'][:n, row].astype(np.float32)                      # (n, L, n_img)
    share = (maps * tmask[None, None]).sum(-1) / np.maximum(maps.sum(-1), 1e-9)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    im = axes[0].imshow(m, aspect='auto', cmap='viridis', vmin=0, vmax=max(0.5, float(m.max())))
    axes[0].set_yticks(range(len(ce.FULL_LAYERS)), [f'L{l}' for l in ce.FULL_LAYERS], fontsize=7)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label('attention mass on image tokens (head mean)', fontsize=7)
    li = list(ce.FULL_LAYERS).index(20)
    axes[1].plot(share[:, li], color=ca.PALETTE[1], lw=1.5, label='L20: share of image attention on the target cell')
    axes[1].plot(m[li], color=ca.PALETTE[0], lw=1.5, label='L20: image mass')
    axes[1].axhline(tmask.mean(), color=ca.GREY, ls=':', lw=1, label='uniform target share')
    axes[1].legend(frameon=False, fontsize=7, ncol=3, loc='upper right')
    axes[1].set_ylim(0, 1)
    marks = [i for i, s in enumerate(toks) if ca.token_class(s) in ('digit', 'colour')]
    for i in marks:
        axes[1].text(i, -0.08, toks[i].strip(), rotation=90, fontsize=6, ha='center', va='top',
                     transform=axes[1].get_xaxis_transform(), color=ca.INK2)
    axes[1].set_xlabel('generated token index (digits and colour words labelled below)', labelpad=22)
    axes[0].set_title(f'{task}: {r["item_a"]}  truth row {r["truth_a"]["row"]}, col {r["truth_a"]["col"]}', loc='left', fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def make_all(run: Path, figures: Path, summary: dict) -> None:
    mech = summary['mech']
    fig_swap(mech, figures / 'channel_swap_answer.png', 'swap', 'final answer')
    fig_swap(mech, figures / 'channel_swap_first_claim.png', 'claims', 'first stated row in the CoT')
    meta = json.loads((run / 'run_mech.json').read_text())
    tok = AutoTokenizer.from_pretrained(meta['runtime']['checkpoint_path'], local_files_only=True) \
        if 'checkpoint_path' in meta['runtime'] else None
    if tok is None:
        from eva.inference import resolve_model
        from utils.project_paths import load_paths
        tok = AutoTokenizer.from_pretrained(resolve_model('Qwen3.5-9B', load_paths()['HF_HUB_CACHE']), local_files_only=True)
    att = ca.attention(run, tok)
    summary['attention'] = {k: v for k, v in att.items() if not k.startswith('_')}
    if summary['attention']:
        fig_attention_summary(summary['attention'], figures / 'attention_summary.png')
        for t in ('rowcol', 'point', 'odd'):
            if (run / f'mech/probes/{t}_b0_clean_A.npz').exists():
                fig_timeline(run, t, tok, figures / f'timeline_{t}.png')
