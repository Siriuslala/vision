"""Experiment 3.13: mechanism-guided fine-tuning of the coordinate readout of Qwen3.5-9B.

Design: docs/analysis/position/design_targeted_finetune.md.

This module holds the parts that do not need a GPU stage driver:

* a minimal LoRA implementation based on forward hooks (no new dependency; the base modules, their
  weights and every analysis hook of experiments 3.11/3.12 stay untouched);
* the parameter-subset definitions of the training conditions (all linear layers, the MLPs of the
  digit-writing layers 28-32, the token mixers of the reading layers 16-24 and a placement control);
* construction of the training and test samples (point / bbox answers in 0-1000 units);
* encoding of a supervised batch (answer-token labels only) and the loss on selected positions;
* coordinate metrics (hit, error, round-number shares, 500 plateau, compression slope, edge snapping).

Layer numbers are 1-based.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------------- answer format

POINT_SUFFIX = (' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].'
                ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.'
                ' Output only the JSON, with no explanation.')
BOX_SUFFIX = (' Give its tight bounding box. Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].'
              ' All coordinates use 0-1000 normalization; x increases rightward and y increases downward.'
              ' Output only the JSON, with no explanation.')
QUERY = 'Locate the object described as: {target}.'
# The base model's own answer template (12466 point/bbox answers of experiment 3.9: 98% use it).
ANSWER_TEMPLATE = '```json\n[\n\t{{"{key}": [{values}], "label": "{label}"}}\n]\n```'


def to_units(v: float, size: float) -> int:
    """Pixel coordinate -> integer 0-1000 unit, clipped."""
    return int(min(1000, max(0, round(1000 * v / size))))


def answer_text(fmt: str, truth: dict, label: str) -> str:
    """Exact-coordinate answer in the base template. truth: {'canvas': [W, H], 'box': [...], 'center': [...]}."""
    W, H = truth['canvas']
    if fmt == 'point':
        vals = [to_units(truth['center'][0], W), to_units(truth['center'][1], H)]
        key = 'point_2d'
    elif fmt == 'bbox':
        b = truth['box']
        vals = [to_units(b[0], W), to_units(b[1], H), to_units(b[2], W), to_units(b[3], H)]
        key = 'bbox_2d'
    else:
        raise ValueError(fmt)
    return ANSWER_TEMPLATE.format(key=key, values=', '.join(map(str, vals)), label=label.replace('"', "'"))


def question(target: str, fmt: str) -> str:
    return QUERY.format(target=target) + (POINT_SUFFIX if fmt == 'point' else BOX_SUFFIX)


# ----------------------------------------------------------------------------- LoRA via forward hooks

MIXER_NAMES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
               'linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.in_proj_b',
               'linear_attn.in_proj_a', 'linear_attn.out_proj')
MLP_NAMES = ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
DELTANET_IN = ('linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.in_proj_b', 'linear_attn.in_proj_a')
FA_NAMES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj')

# Training conditions: which (layer, module) pairs receive a LoRA adapter.
CONDITIONS = {
    'all_linear': {'layers': tuple(range(1, 33)), 'modules': MIXER_NAMES + MLP_NAMES},
    'mlp28_32': {'layers': tuple(range(28, 33)), 'modules': MLP_NAMES},
    'mixer16_24': {'layers': tuple(range(16, 25)), 'modules': MIXER_NAMES},
    'mlp10_14': {'layers': tuple(range(10, 15)), 'modules': MLP_NAMES},     # placement control for mlp28_32
    # Capacity control for (b): same modules, rank 64 (8x the parameters of mlp28_32).
    'mlp28_32_r64': {'layers': tuple(range(28, 33)), 'modules': MLP_NAMES, 'rank': 64},
    # (b') 3.12 round 2: the second digit reads the q0 first-digit margin relayed by the DeltaNet short convolutions of
    # layers 25-31, and the centre prior is written by MLP 27-31: MLP + DeltaNet input projections + conv of layers 25-32.
    'late25_32': {'layers': tuple(range(25, 33)), 'modules': MLP_NAMES + DELTANET_IN + ('linear_attn.conv1d',)},
    # ---- Round 2 (2026-11-03) controls: MLP vs token mixer, and frozen readout layers.
    # Full-attention q/k/v/o only (the 8 full-attention layers; DeltaNet layers have no self_attn module).
    'fa_attn': {'layers': tuple(range(1, 33)), 'modules': FA_NAMES},
    # Gated DeltaNet projections only (in_proj_qkv/z/b/a and out_proj of the 24 DeltaNet layers).
    'dn_proj': {'layers': tuple(range(1, 33)), 'modules': DELTANET_IN + ('linear_attn.out_proj',)},
    # Every MLP, no token mixer.
    'mlp_all': {'layers': tuple(range(1, 33)), 'modules': MLP_NAMES},
    # Full attention + every MLP (all linear layers except the DeltaNet projections).
    'fa_mlp': {'layers': tuple(range(1, 33)), 'modules': FA_NAMES + MLP_NAMES},
    # All linear layers with the digit-writing MLPs of layers 28-32 frozen.
    'all_but_mlp28_32': {'layers': tuple(range(1, 33)), 'modules': MIXER_NAMES + MLP_NAMES,
                         'exclude': (((28, 29, 30, 31, 32), MLP_NAMES),)},
    # All linear layers of layers 1-24: the whole readout stage (layers 25-32, 3.12 / 3.12 round 2) frozen.
    'all_but_25_32': {'layers': tuple(range(1, 25)), 'modules': MIXER_NAMES + MLP_NAMES},
    # Optimisation control for (b): same modules and rank, 10x learning rate.
    'mlp28_32_lr1e3': {'layers': tuple(range(28, 33)), 'modules': MLP_NAMES, 'lr': 1e-3},
    # ---- Round 2 scaled fine-tuning (more natural grounding data + self-distilled instruction data).
    'scaled_all_linear': {'layers': tuple(range(1, 33)), 'modules': MIXER_NAMES + MLP_NAMES,
                          'train_file': 'train_scaled.jsonl'},
    'scaled_all_but_25_32': {'layers': tuple(range(1, 25)), 'modules': MIXER_NAMES + MLP_NAMES,
                             'train_file': 'train_scaled.jsonl'},
    # ---- Round 2 located variants (from the splice analysis: the fix lives in layers 1-4 at the image tokens).
    'early1_4': {'layers': (1, 2, 3, 4), 'modules': MIXER_NAMES + MLP_NAMES},
    # positions 'image': the update is applied only at image-token positions (text tokens see the base weights).
    'early1_4_img': {'layers': (1, 2, 3, 4), 'modules': MIXER_NAMES + MLP_NAMES, 'positions': 'image'},
    'all_linear_img': {'layers': tuple(range(1, 33)), 'modules': MIXER_NAMES + MLP_NAMES, 'positions': 'image'},
    'scaled_early1_4_img': {'layers': (1, 2, 3, 4), 'modules': MIXER_NAMES + MLP_NAMES, 'positions': 'image',
                            'train_file': 'train_scaled.jsonl'},
}


def target_modules(layers: nn.ModuleList, condition: str) -> dict[str, nn.Module]:
    """'L{layer}.{name}' -> nn.Linear (or the DeltaNet nn.Conv1d) for one condition; modules absent in a layer
    type (e.g. DeltaNet projections in full-attention layers) are skipped."""
    spec = CONDITIONS[condition]
    excluded = {(l, n) for ls, ns in spec.get('exclude', ()) for l in ls for n in ns}
    out = {}
    for l in spec['layers']:
        named = dict(layers[l - 1].named_modules())
        for name in spec['modules']:
            if (l, name) in excluded:
                continue
            mod = named.get(name)
            if isinstance(mod, (nn.Linear, nn.Conv1d)):
                out[f'L{l}.{name}'] = mod
    if not out:
        raise ValueError(f'condition {condition} selects no module')
    return out


class LoRA(nn.Module):
    """Low-rank update W + (alpha / r) B A on selected nn.Linear modules, applied by forward hooks.

    A has shape (r, in) with Kaiming-uniform initialisation, B has shape (out, r) and starts at zero, so
    the adapted model equals the base model at step 0. Parameters are float32; the update is computed in
    float32 and added to the bf16 module output. Base weights are never modified, so `.weight` keeps
    referring to the frozen base matrix (relevant for weight-based analyses such as per-head DLA).
    """

    def __init__(self, modules: dict[str, nn.Linear], rank: int, alpha: float):
        super().__init__()
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.names = sorted(modules)
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.D = nn.ParameterDict()
        for n in self.names:
            m = modules[n]
            dev = m.weight.device
            if isinstance(m, nn.Conv1d):
                # Depthwise short convolution (channels, 1, kernel): small enough for a dense delta.
                self.D[self._key(n)] = nn.Parameter(torch.zeros(m.weight.shape, device=dev))
                continue
            a = torch.empty(rank, m.in_features, device=dev)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.A[self._key(n)] = nn.Parameter(a)
            self.B[self._key(n)] = nn.Parameter(torch.zeros(m.out_features, rank, device=dev))
        self._modules_ref = modules
        self._handles: list = []
        self._param: list = []
        # Analysis-only controls (round 2). active: None = every adapted module applies its update, otherwise only
        # the listed module names do. pos_mask: None = every position, otherwise a (B, L) tensor; the update is
        # multiplied by it at each sequence position (0/1 restricts the update to selected tokens).
        self.active: set[str] | None = None
        self.pos_mask: torch.Tensor | None = None
        self._extra_handles: list = []          # e.g. the image-mask pre-hook (removed by detach)

    @staticmethod
    def _key(name: str) -> str:
        return name.replace('.', '__')

    def attach(self) -> None:
        if self._handles or self._param:
            raise RuntimeError('LoRA hooks already attached')
        from torch.nn.utils import parametrize
        for n in self.names:
            if self._key(n) in self.D:
                parametrize.register_parametrization(self._modules_ref[n], 'weight', _AddDelta(self.D[self._key(n)]))
                self._param.append(n)
                continue
            A, B = self.A[self._key(n)], self.B[self._key(n)]

            def hook(module, inputs, output, A=A, B=B, n=n):
                if self.active is not None and n not in self.active:
                    return output
                x = inputs[0]
                delta = (x.float() @ A.t()) @ B.t()
                if self.pos_mask is not None:
                    m = self.pos_mask.to(delta.device, delta.dtype)
                    if m.shape != delta.shape[:2]:
                        raise ValueError(f'pos_mask {tuple(m.shape)} does not match the input {tuple(delta.shape[:2])}')
                    delta = delta * m[..., None]
                return output + (self.scale * delta).to(output.dtype)
            self._handles.append(self._modules_ref[n].register_forward_hook(hook))

    def detach(self) -> None:
        from torch.nn.utils import parametrize
        for h in self._handles + self._extra_handles:
            h.remove()
        self._handles, self._extra_handles = [], []
        self.pos_mask = None
        for n in self._param:
            parametrize.remove_parametrizations(self._modules_ref[n], 'weight', leave_parametrized=False)
        self._param = []

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state(self) -> dict:
        lin = [n for n in self.names if self._key(n) in self.A]
        return {'rank': self.rank, 'alpha': self.alpha, 'names': self.names,
                'A': {n: self.A[self._key(n)].detach().cpu() for n in lin},
                'B': {n: self.B[self._key(n)].detach().cpu() for n in lin},
                'D': {n: self.D[self._key(n)].detach().cpu() for n in self.names if self._key(n) in self.D}}

    def load(self, state: dict) -> None:
        if state['names'] != self.names or state['rank'] != self.rank:
            raise ValueError('Adapter state does not match the condition (module list or rank differ)')
        with torch.no_grad():
            for n in self.names:
                if self._key(n) in self.D:
                    self.D[self._key(n)].copy_(state['D'][n])
                else:
                    self.A[self._key(n)].copy_(state['A'][n])
                    self.B[self._key(n)].copy_(state['B'][n])

    def delta_norms(self) -> dict[str, float]:
        """Frobenius norm of the full update scale * B A per module (computed via the r x r Gram matrices)."""
        out = {}
        with torch.no_grad():
            for n in self.names:
                if self._key(n) in self.D:
                    out[n] = float(self.D[self._key(n)].norm())
                    continue
                A, B = self.A[self._key(n)].float(), self.B[self._key(n)].float()
                out[n] = float(self.scale * torch.sqrt(torch.trace((B.t() @ B) @ (A @ A.t())).clamp_min(0)))
        return out


class _AddDelta(nn.Module):
    """Parametrization W -> W + D with a float32 delta D (the sum is cast back to the weight dtype)."""

    def __init__(self, delta: nn.Parameter):
        super().__init__()
        self.delta = delta

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return (w.float() + self.delta.to(w.device)).to(w.dtype)


def attach_image_mask(model: nn.Module, lora: LoRA, image_token_id: int):
    """Restrict a LoRA's update to image-token positions whenever the multimodal model is called with input_ids.

    A forward pre-hook on the multimodal backbone (model.model) sets lora.pos_mask = (input_ids == image_token_id);
    during cached generation the decoding steps contain no image token, so they use the base weights. Calls with
    inputs_embeds and 4-row MRoPE position ids (analysis passes) take the image positions from the position ids
    (text tokens have equal t / h / w ids; the first image token, whose three ids coincide, is recognised by sharing
    its temporal id with the next image token). Returns the hook handle."""
    def pre(module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is not None:
            lora.pos_mask = (ids == image_token_id).float()
            return
        # inputs_embeds calls (analysis passes): image tokens are the positions whose MRoPE (t, h, w) ids differ;
        # text tokens have t = h = w. position_ids: (4, B, L) with row 0 the sequence index.
        pos = kwargs.get('position_ids')
        if pos is not None and pos.dim() == 3 and pos.shape[0] == 4:
            m = ~((pos[1] == pos[2]) & (pos[2] == pos[3]))
            # The first image token has t = h = w; it shares its temporal id with the image token after it.
            first = torch.zeros_like(m)
            first[:, :-1] = m[:, 1:] & (pos[1][:, :-1] == pos[1][:, 1:])
            lora.pos_mask = (m | first).float()
    handle = model.model.register_forward_pre_hook(pre, with_kwargs=True)
    lora._extra_handles.append(handle)
    return handle


def load_adapter(layers: nn.ModuleList, path: Path, condition: str, model: nn.Module | None = None) -> LoRA:
    """Recreate an adapter from a saved state and attach it. For image-position conditions the image mask hook is
    attached when the multimodal `model` is given (otherwise the caller must set pos_mask)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('condition') != condition:
        raise ValueError(f"Adapter {path} was trained for {state.get('condition')!r}, not {condition!r}")
    lora = LoRA(target_modules(layers, condition), state['rank'], state['alpha'])
    lora.load(state)
    lora.attach()
    if CONDITIONS[condition].get('positions') == 'image' and model is not None:
        attach_image_mask(model, lora, model.config.image_token_id)
    return lora


# ----------------------------------------------------------------------------- samples

@dataclass
class Sample:
    """One supervised or evaluation item.

    family: 'refcoco' | 'flat' | 'texture' | 'grid' (training families) or a test family name.
    image:  absolute path of the input image (sides are multiples of 32).
    fmt:    'point' or 'bbox'.
    truth:  {'canvas': [W, H], 'box': [x0, y0, x1, y1] pixels, 'center': [cx, cy] pixels, ...}.
    prompt: user text after the image.
    label:  value of the "label" field in the training answer (the target expression).
    meta:   free-form; meta['answer_text'] (round-2 instruction samples) replaces the coordinate answer template,
            and such samples use fmt 'text'.
    """
    id: str
    family: str
    split: str
    image: str
    fmt: str
    truth: dict
    prompt: str
    label: str = ''
    meta: dict = field(default_factory=dict)


def dump_samples(path: Path, samples: list[Sample]) -> None:
    path.write_text(''.join(json.dumps(s.__dict__) + '\n' for s in samples))


def read_samples(path: Path) -> list[Sample]:
    return [Sample(**json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


# ----------------------------------------------------------------------------- supervised batches

def encode_batch(processor, prompt_fn, samples: list[Sample], images: list) -> dict:
    """Right-padded batch of prompt + exact answer with labels on the answer tokens only.

    prompt_fn(text) returns the chat-formatted prompt string (generation prompt, thinking off).
    Returns the processor tensors plus 'label_pos' (N,) row index, 'label_idx' (N,) position whose logits
    predict the token, 'label_tok' (N,) target ids, 'label_row' (N,) sample index and 'is_digit' (N,) bool.
    """
    tok = processor.tokenizer
    prompts = [prompt_fn(s.prompt) for s in samples]
    # Instruction samples (round 2) carry their answer text in meta['answer_text']; grounding samples use the template.
    answers = [(s.meta['answer_text'] if 'answer_text' in s.meta else answer_text(s.fmt, s.truth, s.label)) + '<|im_end|>'
               for s in samples]
    tok.padding_side = 'right'
    batch = processor(text=[p + a for p, a in zip(prompts, answers)], images=images, padding=True, return_tensors='pt')
    for image, (_, h, w) in zip(images, batch['image_grid_thw'].tolist()):
        if (h * 16, w * 16) != (image.height, image.width):
            raise ValueError(f'Processor resized a {image.width}x{image.height} image')
    rows, idx, tgt, dig = [], [], [], []
    ids = batch['input_ids']
    lengths = batch['attention_mask'].sum(1).tolist()
    digit_ids = {tok.convert_tokens_to_ids(str(d)) for d in range(10)}
    for b, (p, a) in enumerate(zip(prompts, answers)):
        n_ans = len(tok.encode(a, add_special_tokens=False))
        L = int(lengths[b])
        # The prompt ends with "\n\n" after </think>; the answer starts with "```", so the boundary is a token boundary.
        full = ids[b, :L].tolist()
        if tok.decode(full[L - n_ans:]) != a:
            raise ValueError(f'Answer tokens do not align with the end of sample {samples[b].id}')
        for t in range(L - n_ans, L):
            rows.append(b)
            idx.append(t - 1)
            tgt.append(full[t])
            dig.append(full[t] in digit_ids and 'answer_text' not in samples[b].meta)
    batch['label_row'] = torch.tensor(rows)
    batch['label_idx'] = torch.tensor(idx)
    batch['label_tok'] = torch.tensor(tgt)
    batch['is_digit'] = torch.tensor(dig)
    return batch


def batch_loss(model, batch: dict, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample mean cross-entropy over answer tokens.

    Returns (per-sample mean CE (B,), digit-token CE values (n_digit,), per-token CE (N,)).
    The LM head is applied only at labelled positions, so the (L, vocab) logit tensor is never formed.
    """
    keys = ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids')
    inputs = {k: batch[k].to(device) for k in keys if k in batch}
    out = model.model(**inputs, use_cache=False)
    h = out.last_hidden_state
    rows, idx = batch['label_row'].to(h.device), batch['label_idx'].to(h.device)
    logits = model.lm_head(h[rows, idx]).float()
    ce = nn.functional.cross_entropy(logits, batch['label_tok'].to(logits.device), reduction='none')
    B = int(batch['input_ids'].shape[0])
    sums = torch.zeros(B, device=ce.device).index_add(0, rows.to(ce.device), ce)
    counts = torch.zeros(B, device=ce.device).index_add(0, rows.to(ce.device), torch.ones_like(ce))
    return sums / counts, ce[batch['is_digit'].to(ce.device)], ce


# ----------------------------------------------------------------------------- coordinate metrics

NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def parse_answer(text: str, fmt: str) -> list[float] | None:
    """First point_2d / bbox_2d list in the answer (tolerant to code fences and extra fields)."""
    key, n = ('point_2d', 2) if fmt == 'point' else ('bbox_2d', 4)
    m = re.search(r'"%s"\s*:\s*\[([^\]]*)\]' % key, text)
    if not m:
        return None
    vals = NUM_RE.findall(m.group(1))
    if len(vals) != n:
        return None
    return [float(v) for v in vals]


def box_iou(a, b) -> float:
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.


def score(sample: Sample, text: str) -> dict:
    """Validity, predicted centre (units), hit, centre error (units) and IoU for bbox answers.

    Hit: the predicted centre lies inside the truth box (disc tests: inside the disc).
    """
    coords = parse_answer(text, sample.fmt)
    if coords is None:
        return {'valid': False, 'hit': False}
    W, H = sample.truth['canvas']
    c = np.array(coords[:2]) if sample.fmt == 'point' else (np.array(coords[:2]) + np.array(coords[2:])) / 2
    t = np.array([1000 * sample.truth['center'][0] / W, 1000 * sample.truth['center'][1] / H])
    px = c * np.array([W, H]) / 1000
    if 'diameter' in sample.truth:
        hit = float(np.linalg.norm(px - np.array(sample.truth['center']))) <= sample.truth['diameter'] / 2
    else:
        b = sample.truth['box']
        hit = b[0] <= px[0] <= b[2] and b[1] <= px[1] <= b[3]
    out = {'valid': True, 'coords': coords, 'center': c.tolist(), 'truth_units': t.tolist(), 'hit': bool(hit),
           'error': float(np.linalg.norm(c - t)), 'err_xy': (c - t).tolist()}
    if sample.fmt == 'bbox':
        box_px = (np.array(coords) * np.array([W, H, W, H]) / 1000).tolist()
        out['iou'] = box_iou(box_px, sample.truth['box'])
    return out


def round_share(values: np.ndarray, base: int = 10) -> float:
    """Share of integer coordinate values that are multiples of `base` (0.1 expected for base 10 if uniform)."""
    v = np.round(np.asarray(values)).astype(int)
    return float((v % base == 0).mean()) if len(v) else float('nan')


def plateau_share(pred: np.ndarray, truth: np.ndarray, lo: float = 400, hi: float = 610) -> tuple[float, int]:
    """Share of outputs in [480, 520] among truths in [lo, hi) (3.12 definition). Returns (share, n)."""
    m = (truth >= lo) & (truth < hi)
    if not m.any():
        return float('nan'), 0
    return float(((pred[m] >= 480) & (pred[m] <= 520)).mean()), int(m.sum())


def slope(pred: np.ndarray, truth: np.ndarray, lo: float = 0, hi: float = 1000) -> float:
    m = (truth >= lo) & (truth < hi)
    return float(np.polyfit(truth[m], pred[m], 1)[0]) if m.sum() > 2 else float('nan')


def edge_share(pred: np.ndarray, edges: list[np.ndarray]) -> tuple[float, float]:
    """Grid snapping: share of outputs closer to a cell edge than to a cell centre, and median edge distance
    in cell units. edges[i]: (n+1,) column edges (units) of item i."""
    closer, dist = [], []
    for p, e in zip(pred, edges):
        w = float(np.mean(np.diff(e)))
        de = float(np.min(np.abs(e - p)))
        centres = (e[:-1] + e[1:]) / 2
        dc = float(np.min(np.abs(centres - p)))
        closer.append(de < dc)
        dist.append(de / w)
    return float(np.mean(closer)), float(np.median(dist))
