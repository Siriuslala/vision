"""Repeated-object layout controls and token-exclusion readouts for Qwen3.5."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

from analysis_tools.distance_register import synthetic, unique_top
from analysis_tools.visual_detail import spatial_grid
from eva.inference import resolve_model
from utils.project_paths import load_paths


def layouts() -> list[dict]:
    """Fix layouts before inspecting any features, with integer patch-aligned windows."""
    cases = []
    for rows, cell in [(6, 64), (6, 128), (20, 32), (20, 64)]:
        boxes = [(c*cell, r*cell, (c+1)*cell, (r+1)*cell) for r in range(rows) for c in range(rows)]
        cases.append({'name': f'grid{rows}_cell{cell}', 'size': rows*cell, 'cell': cell, 'boxes': boxes,
                      'layout': 'grid', 'rows': rows})
    boxes = []
    for r in range(6):
        for c in range(6):
            # Vary blank gaps by whole merged patches; every local object raster is identical.
            x, y = c*96 + (r%2)*32, r*96
            boxes.append((x, y, x+64, y+64))
    cases.append({'name': 'staggered36', 'size': 608, 'cell': 64, 'boxes': boxes, 'layout': 'staggered'})
    boxes = []
    for i in range(16):
        angle = 2*math.pi*i/16
        x = round((384+280*math.cos(angle)-32)/32)*32
        y = round((384+280*math.sin(angle)-32)/32)*32
        boxes.append((x, y, x+64, y+64))
    cases.append({'name': 'circle16', 'size': 768, 'cell': 64, 'boxes': boxes, 'layout': 'circle'})
    return cases


def render(layout: dict, family: str, target: int | None = None) -> Image.Image:
    """Resize complete object cells identically, without antialiasing or target shifts."""
    base = synthetic(family, 0).crop((0, 0, 64, 64))
    edited = synthetic(family, 0, 0).crop((0, 0, 64, 64))
    image = Image.new('RGB', (layout['size'], layout['size']), 'white')
    for i, box in enumerate(layout['boxes']):
        tile = edited if i == target else base
        image.paste(tile.resize((layout['cell'], layout['cell']), Image.Resampling.NEAREST), box[:2])
    return image


def region_indices(grid_shape: tuple[int, int], image_size: int, boxes: list) -> torch.Tensor:
    """Return aligned relative patch positions in every fixed object window."""
    h, w = grid_shape
    indices = []
    for x0, y0, x1, y1 in boxes:
        edges = np.array([x0*w/image_size, y0*h/image_size, x1*w/image_size, y1*h/image_size])
        if not np.allclose(edges, np.round(edges), atol=1e-7):
            raise ValueError('Object windows must align with the feature grid.')
        a, b, c, d = np.round(edges).astype(int)
        indices.append(torch.tensor([r*w+col for r in range(b, d) for col in range(a, c)]))
    if len({len(v) for v in indices}) != 1 or len(indices[0]) == 0:
        raise ValueError('Object windows must contain equal nonzero numbers of tokens.')
    return torch.stack(indices)


def pairwise_scores(features: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compare pairs on the same retained relative positions, then average valid distances."""
    n, p, d = features.shape
    if valid.shape != (n, p) or n < 2:
        raise ValueError('Expected a validity mask per object and relative patch.')
    dot = torch.zeros(n, n)
    aa, bb, count = torch.zeros_like(dot), torch.zeros_like(dot), torch.zeros_like(dot)
    for j in range(p):
        v = features[:, j]
        keep = valid[:, j, None] & valid[None, :, j]
        norm = v.square().sum(-1)
        dot += (v @ v.T)*keep
        aa += norm[:, None]*keep
        bb += norm[None, :]*keep
        count += keep
    supported = count > 0
    supported.fill_diagonal_(False)
    rms = ((aa+bb-2*dot).clamp_min(0)/(count*d).clamp_min(1)).sqrt()
    cosine = (1-dot/(aa*bb).clamp_min(1e-30).sqrt()).clamp_min(0)
    result = {}
    for name, distance in [('rms', rms), ('cosine', cosine)]:
        score = (distance*supported).sum(-1)/supported.sum(-1).clamp_min(1)
        score[supported.sum(-1) == 0] = -torch.inf
        result[name] = score
    return result


def readouts(grid: torch.Tensor, indices: torch.Tensor) -> tuple[dict, dict]:
    """Exclude high-norm or random tokens only from analysis, never from model attention."""
    flat = grid.reshape(-1, grid.shape[-1]).float()
    norm = flat.norm(dim=-1)
    k = math.ceil(.05*len(flat))
    choices = {'all': torch.empty(0, dtype=torch.long), 'exclude_high5': torch.argsort(norm, stable=True)[-k:]}
    for seed in range(3):
        choices[f'exclude_random5_seed{seed}'] = torch.randperm(len(flat), generator=torch.Generator().manual_seed(seed))[:k]
    scores, support = {}, {}
    for label, excluded in choices.items():
        keep = torch.ones(len(flat), dtype=torch.bool)
        keep[excluded] = False
        valid = keep[indices]
        values = pairwise_scores(flat[indices], valid)
        support[label] = {'excluded_image_tokens': len(excluded), 'total_image_tokens': len(flat),
                          'empty_objects': int((valid.sum(-1) == 0).sum()),
                          'object_token_counts': valid.sum(-1).tolist()}
        for metric, value in values.items():
            finite = torch.isfinite(value)
            peak = unique_top(value) if finite.any() else None
            scores[f'{label}_{metric}'] = {'peak': peak, 'scores': [float(v) if math.isfinite(float(v)) else None for v in value]}
    return scores, support


def chosen_targets(n: int, count: int | None = None) -> list[int]:
    """Fix four boundary indices plus random positions, independent of model activations."""
    count = min(n, count if count is not None else (24 if n > 100 else 12))
    if count < 4:
        raise ValueError('At least four targets are required.')
    side = int(round(math.sqrt(n)))
    anchors = {0, n-1, side-1, n-side} if side*side == n else {0, n//3, 2*n//3, n-1}
    pool = np.array([i for i in range(n) if i not in anchors])
    selected = np.random.default_rng(20261001).permutation(pool)[:count-len(anchors)]
    return sorted(anchors | set(selected.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', default='Qwen3.5-9B')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--layouts', nargs='+', default=None)
    parser.add_argument('--targets-per-case', type=int, default=None)
    args = parser.parse_args()
    selected_layouts = [item for item in layouts() if args.layouts is None or item['name'] in args.layouts]
    if args.layouts is not None and {item['name'] for item in selected_layouts} != set(args.layouts):
        raise ValueError('Unknown layout selection.')
    if args.targets_per_case is not None and args.targets_per_case < 4:
        raise ValueError('At least four targets per case are required.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(0)
    torch.set_num_threads(4)
    checkpoint = resolve_model(args.model, load_paths()['HF_HUB_CACHE'])
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16,
                  device_map='cuda:0', attn_implementation='sdpa').eval()
    captured, mask = {}, None
    def vision_hook(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().float().cpu()
        return hook
    def decoder_hook(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value[0, mask].detach().float().cpu()
        return hook
    handles = [model.model.visual.patch_embed.register_forward_hook(vision_hook('patch_embed'))]
    handles += [model.model.visual.blocks[i-1].register_forward_hook(vision_hook(f'vit_{i:02}')) for i in [3,9,18,27]]
    handles += [model.model.language_model.layers[i-1].register_forward_hook(decoder_hook(f'decoder_{i:02}')) for i in [1,4,8,16,24,28,31,32]]
    chat = [{'role':'user', 'content':[{'type':'image'}, {'type':'text','text':'Describe the image.'}]}]
    prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    metadata = {'model': args.model, 'checkpoint': checkpoint.name, 'torch':torch.__version__, 'transformers':transformers.__version__,
                'dtype':'bfloat16', 'attention':'sdpa', 'thinking':'off', 'prompt':'Describe the image.', 'seed':0,
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'target_sampling_seed':20261001, 'target_lists':{item['name']:chosen_targets(len(item['boxes']),args.targets_per_case) for item in selected_layouts},
                'start_utc':datetime.now(timezone.utc).isoformat(), 'layouts':selected_layouts, 'targets_per_case':args.targets_per_case,
                'decoder_subset':'normal and five sorted-target quantiles per layout/family', 'random_exclusion_seeds':[0,1,2]}
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    started = time.time()
    records, normal_records = [], []
    with torch.inference_mode():
        for layout in selected_layouts:
            for family in ['person','ring','square']:
                case = f'{layout["name"]}_{family}'
                folder = args.output/case
                folder.mkdir()
                targets = chosen_targets(len(layout['boxes']), args.targets_per_case)
                decoder_targets = {None} | {targets[i] for i in np.linspace(0,len(targets)-1,5).round().astype(int)}
                baselines, normal_peaks = {}, {}
                for target in [None]+targets:
                    image = render(layout, family, target)
                    image.save(folder/f'{"normal" if target is None else target}.png')
                    batch = processor(text=[prompt], images=[image], return_tensors='pt').to('cuda:0')
                    _, h, w = batch.image_grid_thw[0].tolist()
                    if (h*16,w*16) != image.size[::-1]:
                        raise ValueError('Unexpected processor resize; geometry no longer matches protocol.')
                    mask = batch.input_ids[0] == model.config.image_token_id
                    captured.clear()
                    if target in decoder_targets:
                        # The decoder receives exactly the same frozen visual encoder outputs as vision-only cases.
                        merger_values = []
                        handle = model.model.visual.merger.register_forward_hook(lambda m, i, o: merger_values.append(o.detach().float().cpu()))
                        model.model(**batch, use_cache=False)
                        handle.remove()
                        merger = merger_values[0]
                    else:
                        out = model.model.visual(batch.pixel_values, grid_thw=batch.image_grid_thw)
                        merger = out.pooler_output.detach().float().cpu()
                    grids = {name: value.reshape(h//2,w//2,-1) if name.startswith('decoder') else spatial_grid(value,h,w)
                             for name,value in captured.items()}
                    grids['merger'] = merger.reshape(h//2,w//2,-1)
                    for stage, grid in grids.items():
                        indices = region_indices(grid.shape[:2], layout['size'], layout['boxes'])
                        scores, support = readouts(grid, indices)
                        if target is None:
                            baselines[stage] = grid
                            normal_peaks[stage] = {k:v['peak'] for k,v in scores.items()}
                            norm = grid.norm(dim=-1)
                            normal_records.append({'case':case,'stage':stage,'grid':list(grid.shape[:2]),
                                'norm_max_over_median':float(norm.max()/norm.median()), 'norm_map':norm.tolist(),
                                'scores':scores,'support':support})
                        else:
                            base = baselines[stage]
                            delta = (grid-base).square().mean(-1).sqrt().flatten()[indices].mean(-1)
                            cosine = (1-torch.nn.functional.cosine_similarity(grid,base,dim=-1)).clamp_min(0).flatten()[indices].mean(-1)
                            records.append({'case':case,'stage':stage,'target':target,
                                'paired_rms_correct':unique_top(delta)==target, 'paired_cosine_correct':unique_top(cosine)==target,
                                'scores':{k:{'peak':v['peak'],'correct':v['peak']==target,
                                            'same_normal_peak':v['peak'] is not None and v['peak']==normal_peaks[stage][k]} for k,v in scores.items()},
                                'support':{k:{kk:vv for kk,vv in val.items() if kk != 'object_token_counts'} for k,val in support.items()}})
                            if target == targets[len(targets)//2]:
                                (folder/f'example_{stage}.json').write_text(json.dumps({'target':target,'scores':scores,'norm_map':grid.norm(dim=-1).tolist()}))
                    print(f'{case} target={target} elapsed={time.time()-started:.1f}s',flush=True)
                (args.output/'results.json').write_text(json.dumps({'metadata':metadata,'elapsed_seconds':time.time()-started,'normals':normal_records,'records':records}))
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()
