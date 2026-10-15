"""Shared runtime for frozen Qwen3.5 / Qwen3-VL analysis experiments.

Responsibilities: load a checkpoint offline, build single-turn image+text inputs
without hidden resizing, locate image tokens, read first-token logits, run greedy
generation, and capture or replace decoder-layer outputs through forward hooks.
Layer numbers in this module are 1-based, matching the analysis documents.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from eva.inference import resolve_model
from utils.project_paths import load_paths


@dataclass
class Runtime:
    name: str
    checkpoint: Path
    model: torch.nn.Module
    processor: object
    device: str

    @property
    def family(self) -> str:
        return self.model.config.model_type          # 'qwen3_5' or 'qwen3_vl'

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def layers(self) -> torch.nn.ModuleList:
        return self.model.model.language_model.layers


def load(name: str, device: str = 'cuda:0', attn: str = 'sdpa', dtype=torch.bfloat16) -> Runtime:
    checkpoint = resolve_model(name, load_paths()['HF_HUB_CACHE'])
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, local_files_only=True, dtype=dtype, device_map=device, attn_implementation=attn).eval()
    model.requires_grad_(False)
    return Runtime(name, checkpoint, model, processor, device)


def describe(rt: Runtime) -> dict:
    """Provenance fields every experiment should store in its metadata."""
    return {'model': rt.name, 'checkpoint_revision': rt.checkpoint.name, 'family': rt.family,
            'dtype': str(rt.model.dtype), 'attention': rt.model.config._attn_implementation,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'runtime_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def prompt_text(rt: Runtime, n_images: int, text: str, thinking: bool = False) -> str:
    content = [{'type': 'image'} for _ in range(n_images)] + [{'type': 'text', 'text': text}]
    chat = [{'role': 'user', 'content': content}]
    # Qwen3-VL Instruct has no thinking switch; the keyword is ignored by its template.
    return rt.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=thinking)


def build(rt: Runtime, images: list[Image.Image], text: str, thinking: bool = False):
    """Tokenize one conversation turn and verify the processor kept every image size."""
    batch = rt.processor(text=[prompt_text(rt, len(images), text, thinking)], images=images or None,
                         return_tensors='pt').to(rt.device)
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist() if images else []):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image to {w * 16}x{h * 16}; '
                             'stimulus geometry would no longer match the ground truth.')
    return batch


def image_spans(rt: Runtime, batch) -> list[dict]:
    """Sequence positions of each image's merged tokens and its merged grid (rows, cols)."""
    ids = batch['input_ids'][0]
    positions = (ids == rt.model.config.image_token_id).nonzero().flatten().tolist()
    grids = batch['image_grid_thw'].tolist()
    spans, cursor = [], 0
    for _, h, w in grids:
        count = (h // 2) * (w // 2)
        run = positions[cursor:cursor + count]
        if len(run) != count or run[-1] - run[0] != count - 1:
            raise ValueError('Image tokens are not contiguous or do not match image_grid_thw.')
        spans.append({'start': run[0], 'end': run[-1] + 1, 'grid': (h // 2, w // 2)})
        cursor += count
    if cursor != len(positions):
        raise ValueError('Unassigned image tokens remain in the sequence.')
    return spans


def token_id(rt: Runtime, word: str) -> int:
    ids = rt.tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f'{word!r} is not a single token: {ids}')
    return ids[0]


@torch.inference_mode()
def next_token_logits(rt: Runtime, batch) -> torch.Tensor:
    """Float32 logits of the first answer token (prefill only)."""
    return rt.model(**batch).logits[0, -1].float()


def yes_no(rt: Runtime, batch) -> dict:
    """Discrimination score s = z_Yes - z_No at the first answer token."""
    logits = next_token_logits(rt, batch)
    z_yes, z_no = logits[token_id(rt, 'Yes')].item(), logits[token_id(rt, 'No')].item()
    top = int(logits.argmax())
    return {'s': z_yes - z_no, 'z_yes': z_yes, 'z_no': z_no, 'top_token': rt.tokenizer.decode([top])}


@torch.inference_mode()
def generate(rt: Runtime, batch, max_new_tokens: int) -> dict:
    """Greedy decoding; truncation is reported, never hidden."""
    out = rt.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                            temperature=None, top_p=None, top_k=None)
    new = out[0, batch['input_ids'].shape[1]:]
    eos_ids = rt.model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    finished = len(new) > 0 and int(new[-1]) in eos_ids
    return {'text': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': int(len(new)),
            'finish_reason': 'eos' if finished else 'length'}


@torch.inference_mode()
def generate_batch(rt: Runtime, items: list[tuple[list[Image.Image], str]], max_new_tokens: int,
                   thinking: bool = False) -> list[dict]:
    """Greedy decoding of several single-turn conversations in one left-padded batch.

    Bare-answer decoding at batch size 1 is bound by per-step overhead, so batching is
    the main throughput lever. Callers should still spot-check equivalence with
    `generate` on a subset, since bf16 kernels may round differently with padding.
    """
    texts = [prompt_text(rt, len(images), text, thinking) for images, text in items]
    images = [image for imgs, _ in items for image in imgs]
    rt.processor.tokenizer.padding_side = 'left'
    batch = rt.processor(text=texts, images=images or None, padding=True, return_tensors='pt').to(rt.device)
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist() if images else []):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image to {w * 16}x{h * 16}.')
    out = rt.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                            temperature=None, top_p=None, top_k=None)
    eos_ids = rt.model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    pad_id = rt.tokenizer.pad_token_id
    results = []
    for row in out[:, batch['input_ids'].shape[1]:]:
        ids = row.tolist()
        end = next((i + 1 for i, t in enumerate(ids) if t in eos_ids), None)
        new = ids[:end] if end is not None else [t for t in ids if t != pad_id]
        results.append({'text': rt.tokenizer.decode(new, skip_special_tokens=True), 'n_tokens': len(new),
                        'finish_reason': 'eos' if end is not None else 'length'})
    return results


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def capture(rt: Runtime, layers: list[int], positions: torch.Tensor | None = None, to_cpu: bool = True):
    """Record decoder-layer outputs (1-based layers) from the prefill pass.

    Yields a dict layer -> tensor of shape (len(positions) or seq_len, hidden).
    """
    store, handles = {}, []
    for layer in layers:
        def hook(module, inputs, output, layer=layer):
            h = _hidden(output)[0]
            if layer in store:        # later generation steps are ignored
                return
            value = h if positions is None else h[positions]
            value = value.detach().float()
            store[layer] = value.cpu() if to_cpu else value
        handles.append(rt.layers[layer - 1].register_forward_hook(hook))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch(rt: Runtime, replacements: dict[int, tuple[torch.Tensor, torch.Tensor]]):
    """Replace decoder-layer outputs at given sequence positions during the prefill pass.

    replacements: layer (1-based) -> (positions LongTensor of shape (k,), values of shape (k, hidden)).
    The hook only fires when the full prompt is present, so cached decoding steps are untouched.
    """
    handles = []
    for layer, (positions, values) in replacements.items():
        def hook(module, inputs, output, positions=positions, values=values):
            h = _hidden(output)
            if h.shape[1] <= int(positions.max()):
                return output
            h[0, positions.to(h.device)] = values.to(h.device, h.dtype)
            return output
        handles.append(rt.layers[layer - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
