"""GPU engine for experiment 1.7: greedy CoT decoding with decode-time interventions and attention probes.

The engine runs its own greedy loop instead of `generate`, so that the cache built by the prefill can be
edited between prefill and decoding:

* prefill covers prompt tokens [0, P-1); the last prompt token is fed as the first decode step, so every
  generated token (including the first) is computed from the possibly edited cache;
* image-KV swap: the keys/values of the image positions in all 8 full-attention layers are replaced by
  those of a donor prompt with identical token layout (another image);
* text-KV swap: the same for the post-image prompt positions [img_end, P-1);
* DeltaNet swap: the recurrent and convolution states of all 24 Gated DeltaNet layers are replaced by
  the donor's end-of-prefill states;
* image mask: during decoding the full-attention layers may not attend to image positions at all.

Attention probes record, for every decode step and every full-attention layer, the attention mass per
head on image / prompt-text / generated positions and the head-mean attention map over image tokens.
Probes compute softmax(q k^T / sqrt(d) + mask) from the same q, k the SDPA kernel receives; the model
output itself is always produced by the unmodified SDPA kernel.

Batches must have identical prompt lengths (no padding); stimuli of one task share image size and prompt.
Shared modules are used read-only; nothing here changes their behaviour.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

from analysis_tools.common import qwen_runtime as qr
from eva.inference import resolve_model
from utils.project_paths import load_paths

FULL_LAYERS = tuple(range(4, 33, 4))          # 1-based full-attention layers of Qwen3.5-9B
LINEAR_LAYERS = tuple(l for l in range(1, 33) if l not in FULL_LAYERS)


def load_split(name: str, max_memory: dict, attn: str = 'sdpa') -> qr.Runtime:
    """Load with device_map='auto' across several GPUs (max_memory like {0: '12GiB', 7: '12GiB'}).

    Used only when no single GPU has room for the whole model; the Runtime fields match qwen_runtime.load.
    """
    checkpoint = resolve_model(name, load_paths()['HF_HUB_CACHE'])
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16, device_map='auto', max_memory=max_memory,
        attn_implementation=attn).eval()
    model.requires_grad_(False)
    device = str(model.model.language_model.embed_tokens.weight.device)
    return qr.Runtime(name, checkpoint, model, processor, device)


# ----------------------------------------------------------------------------- inputs

def build_batch(rt: qr.Runtime, images: list, texts: list[str], thinking: bool = False, prefixes: list[str] | None = None):
    """Unpadded batch of single-image conversations; all rows must tokenize to the same length."""
    prefixes = prefixes or [''] * len(texts)
    full = [qr.prompt_text(rt, 1, t, thinking) + p for t, p in zip(texts, prefixes)]
    batch = rt.processor(text=full, images=images, padding=False, return_tensors='pt')
    lengths = {len(ids) for ids in batch['input_ids']} if isinstance(batch['input_ids'], list) else {batch['input_ids'].shape[1]}
    if len(lengths) != 1:
        raise ValueError(f'rows have different prompt lengths {lengths}; split the batch')
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image.')
    return batch.to(rt.device)


def spans(rt: qr.Runtime, batch) -> dict:
    """Image span and prompt length (identical for all rows)."""
    ids = batch['input_ids']
    img = (ids[0] == rt.model.config.image_token_id).nonzero().flatten()
    for row in ids[1:]:
        if not torch.equal((row == rt.model.config.image_token_id).nonzero().flatten(), img):
            raise ValueError('image positions differ across rows')
    return {'img_start': int(img[0]), 'img_end': int(img[-1]) + 1, 'P': ids.shape[1]}


# ----------------------------------------------------------------------------- attention probe

@dataclass
class Probe:
    """Per decode step records; filled by the patched attention function."""
    sp: dict
    record: bool = True
    mask_image: bool = False
    steps: list = field(default_factory=list)       # list of dict layer -> tensors for the current step
    _current: dict = field(default_factory=dict)

    def new_step(self):
        if self._current:
            self.steps.append(self._current)
        self._current = {}

    def finish(self):
        self.new_step()


_ACTIVE: Probe | None = None
_ORIG_SDPA = mq.ALL_ATTENTION_FUNCTIONS['sdpa']


def _probe_attention(module, query, key, value, attention_mask, **kwargs):
    probe = _ACTIVE
    is_decoder_full = isinstance(module, mq.Qwen3_5Attention)
    if probe is not None and is_decoder_full and query.shape[2] == 1:
        sp = probe.sp
        if probe.mask_image:
            T = key.shape[2]
            add = torch.zeros(query.shape[0], 1, 1, T, device=query.device, dtype=query.dtype)
            add[..., sp['img_start']:sp['img_end']] = torch.finfo(query.dtype).min
            attention_mask = add if attention_mask is None else attention_mask[..., :T] + add
        if probe.record:
            layer = module.layer_idx + 1
            groups = query.shape[1] // key.shape[1]
            k = key.repeat_interleave(groups, dim=1)
            logits = torch.matmul(query.float(), k.float().transpose(-1, -2)) * module.scaling   # (B, H, 1, T)
            if attention_mask is not None:
                logits = logits + attention_mask[..., :logits.shape[-1]].float()
            w = logits.softmax(-1)[:, :, 0]                                                       # (B, H, T)
            s0, s1, P = sp['img_start'], sp['img_end'], sp['P']
            probe._current[layer] = {
                'img': w[..., s0:s1].sum(-1).cpu(),                  # (B, H)
                'pre': w[..., :s0].sum(-1).cpu(),                    # system / template tokens before the image
                'txt': w[..., s1:P].sum(-1).cpu(),                   # question text and template after the image
                'gen': w[..., P:].sum(-1).cpu(),                     # generated tokens (incl. the current one)
                'map': w[..., s0:s1].mean(1).to(torch.float16).cpu(),   # (B, n_img) head-mean
                'map_h11': w[:, 10, s0:s1].to(torch.float16).cpu() if layer == 20 else None,   # L20 H11 (1-based head)
            }
    return _ORIG_SDPA(module, query, key, value, attention_mask, **kwargs)


@contextmanager
def probing(probe: Probe | None):
    global _ACTIVE
    mq.ALL_ATTENTION_FUNCTIONS['sdpa'] = _probe_attention
    _ACTIVE = probe
    try:
        yield probe
    finally:
        _ACTIVE = None
        del mq.ALL_ATTENTION_FUNCTIONS['sdpa']


# ----------------------------------------------------------------------------- prefill and cache edits

@torch.inference_mode()
def prefill(rt: qr.Runtime, batch):
    """Run the prompt without its last token; return the cache and the 3-axis position offset."""
    P = batch['input_ids'].shape[1]
    kw = {k: batch[k] for k in ('pixel_values', 'image_grid_thw') if k in batch}
    if 'mm_token_type_ids' in batch:
        kw['mm_token_type_ids'] = batch['mm_token_type_ids'][:, :P - 1]
    out = rt.model(input_ids=batch['input_ids'][:, :P - 1], use_cache=True, logits_to_keep=1, **kw)
    delta = rt.model.model.rope_deltas.clone()          # (B, 1)
    return out.past_key_values, delta


def _layer(cache, layer: int):
    return cache.layers[layer - 1]


def clone_cache(cache):
    import copy
    return copy.deepcopy(cache)


def swap_into(cache, donor, sp: dict, channels: set[str], rows: torch.Tensor | None = None) -> None:
    """Copy the donor's cache entries for the requested channels into `cache` (in place).

    channels: subset of {'img', 'txt', 'delta'}. rows: optional batch rows to edit (default all).
    """
    sel = slice(None) if rows is None else rows
    s0, s1, P = sp['img_start'], sp['img_end'], sp['P']
    for layer in FULL_LAYERS:
        a, b = _layer(cache, layer), _layer(donor, layer)
        if 'img' in channels:
            a.keys[sel, :, s0:s1] = b.keys[sel, :, s0:s1]
            a.values[sel, :, s0:s1] = b.values[sel, :, s0:s1]
        if 'txt' in channels:
            a.keys[sel, :, s1:P - 1] = b.keys[sel, :, s1:P - 1]
            a.values[sel, :, s1:P - 1] = b.values[sel, :, s1:P - 1]
    if 'delta' in channels:
        for layer in LINEAR_LAYERS:
            a, b = _layer(cache, layer), _layer(donor, layer)
            a.recurrent_states[0][sel] = b.recurrent_states[0][sel]
            a.conv_states[0][sel] = b.conv_states[0][sel]


# ----------------------------------------------------------------------------- decoding

@torch.inference_mode()
def decode(rt: qr.Runtime, cache, delta: torch.Tensor, first_ids: torch.Tensor, start: int, max_new: int,
           probe: Probe | None = None, forced: list[list[int]] | None = None) -> list[dict]:
    """Greedy decoding from an edited cache.

    first_ids: (B,) last prompt token; start: its absolute position index (P-1).
    forced: optional per-row token lists to teacher-force before free decoding (not used by default).
    Returns per row {'ids', 'text', 'finish_reason', 'n_tokens'}; probe steps align with generated tokens
    (step t is the forward pass that produces token t, with the token t-1 as input).
    """
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    B = first_ids.shape[0]
    cur = first_ids.view(B, 1)
    out_ids = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool)
    with probing(probe):
        for t in range(max_new):
            pos = torch.full((3, B, 1), start + t, device=cur.device, dtype=torch.long) + delta.view(1, B, 1).to(cur.device)
            if probe is not None:
                probe.new_step()
            logits = rt.model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True,
                              logits_to_keep=1).logits[:, -1]
            nxt = logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    tok = int(nxt[b])
                    out_ids[b].append(tok)
                    if tok in eos:
                        done[b] = True
            if bool(done.all()):
                break
            cur = nxt.view(B, 1)
        if probe is not None:
            probe.finish()
    res = []
    for b in range(B):
        ids = out_ids[b]
        fin = bool(ids) and ids[-1] in eos
        res.append({'ids': ids, 'text': rt.tokenizer.decode(ids, skip_special_tokens=True),
                    'finish_reason': 'eos' if fin else 'length', 'n_tokens': len(ids)})
    return res


def run_condition(rt, batch_a, cache_a, delta, sp, max_new: int, donor=None, channels: set[str] = frozenset(),
                  mask_image: bool = False, record: bool = False):
    """Decode batch A under one intervention; cache_a is left untouched (a copy is edited)."""
    cache = clone_cache(cache_a)
    if donor is not None and channels:
        swap_into(cache, donor, sp, set(channels))
    probe = Probe(sp, record=record, mask_image=mask_image) if (record or mask_image) else None
    first = batch_a['input_ids'][:, sp['P'] - 1]
    res = decode(rt, cache, delta, first, sp['P'] - 1, max_new, probe)
    del cache
    return res, probe


# ----------------------------------------------------------------------------- left-padded continuation (faithfulness tests)

@torch.inference_mode()
def continue_batch(rt: qr.Runtime, items: list[tuple], max_new: int, thinking: bool = False) -> list[dict]:
    """Greedy continuation of (image, prompt, assistant_prefix) triples in one left-padded batch.

    Mirrors qwen_runtime.generate_batch, with an assistant prefix appended after the chat template.
    """
    texts = [qr.prompt_text(rt, 1, p, thinking) + prefix for _, p, prefix in items]
    images = [im for im, _, _ in items]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images, padding=True, return_tensors='pt').to(rt.device)
    out = rt.model.generate(**batch, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
    eos = rt.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    pad = rt.tokenizer.pad_token_id
    res = []
    for row in out[:, batch['input_ids'].shape[1]:]:
        ids = row.tolist()
        end = next((i + 1 for i, t in enumerate(ids) if t in eos), None)
        new = ids[:end] if end is not None else [t for t in ids if t != pad]
        res.append({'text': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': len(new),
                    'finish_reason': 'eos' if end is not None else 'length'})
    return res
