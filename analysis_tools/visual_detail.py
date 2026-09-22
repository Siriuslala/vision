"""Training-free sensitivity and within-image outlier probes for Qwen3.5."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors import safe_open
import transformers
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from eva.inference import resolve_model
from utils.project_paths import load_paths


def spatial_grid(features: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Undo the processor's 2x2 block-major order for unmerged features."""
    if height % 2 or width % 2 or features.shape[0] != height * width:
        raise ValueError('Expected an even spatial grid and one feature per patch.')
    return features.reshape(height // 2, width // 2, 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(height, width, -1)


def cell_features(grid: torch.Tensor, rows: int = 6) -> torch.Tensor:
    """Keep every relative patch position when comparing repeated figures."""
    h, w, d = grid.shape
    if h % rows or w % rows:
        raise ValueError('Figure boundaries must align with the feature grid.')
    return grid.reshape(rows, h // rows, rows, w // rows, d).permute(0, 2, 1, 3, 4).reshape(rows * rows, -1)


def figure_image(target: int | None = None, gap: int = 0) -> Image.Image:
    """Render a lossless 6x6 grid; remove a gap from one figure's right arm."""
    im = Image.new('RGB', (384, 384), 'white')
    draw = ImageDraw.Draw(im)
    for i in range(36):
        x, y = (i % 6) * 64, (i // 6) * 64
        draw.ellipse((x + 24, y + 8, x + 40, y + 24), outline='black', width=2)
        draw.line((x + 32, y + 24, x + 32, y + 44), fill='black', width=2)
        draw.line((x + 16, y + 32, x + 48, y + 32), fill='black', width=2)
        draw.line((x + 32, y + 44, x + 20, y + 57), fill='black', width=2)
        draw.line((x + 32, y + 44, x + 44, y + 57), fill='black', width=2)
        if i == target and gap:
            draw.rectangle((x + 39, y + 32, x + 38 + gap, y + 33), fill='white')
    return im


def load_vision(model_path: Path, device: str) -> Qwen3_5VisionModel:
    """Load only visual weights, rejecting incomplete or unexpected tensors."""
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if cfg.model_type != 'qwen3_5':
        raise ValueError('This experiment requires a Qwen3.5 checkpoint.')
    cfg.vision_config._attn_implementation = 'sdpa'
    model = Qwen3_5VisionModel(cfg.vision_config)
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.visual.'
    weights = {}
    for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
        with safe_open(model_path / shard, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    weights[key.removeprefix(prefix)] = handle.get_tensor(key)
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


def heatmap(values: torch.Tensor, size: int = 240) -> Image.Image:
    """Use nearest-neighbor display to avoid inventing sub-patch precision."""
    a = values.detach().cpu().numpy()
    a = a / max(float(a.max()), 1e-12)
    rgb = np.stack([255 * a, 70 * a, 255 * (1 - a)], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen3.5-9B')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = load_paths()
    model_path = resolve_model(args.model, paths['HF_HUB_CACHE'])
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_vision(model_path, args.device)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    selected = [3, 9, 18, 27]
    captured = {}
    handles = []
    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().float().cpu()
        return hook
    handles.append(model.patch_embed.register_forward_hook(capture('patch_embed')))
    for layer in selected:
        handles.append(model.blocks[layer - 1].register_forward_hook(capture(f'layer_{layer}')))
    handles.append(model.merger.norm.register_forward_hook(capture('merger_norm')))

    @torch.inference_mode()
    def encode(im):
        batch = processor(images=im, return_tensors='pt')
        t, h, w = batch['image_grid_thw'][0].tolist()
        if (t, h, w) != (1, 24, 24):
            raise ValueError(f'Unexpected resizing: {(t, h, w)}; expected unchanged 384x384.')
        out = model(batch['pixel_values'].to(args.device, torch.float32), grid_thw=batch['image_grid_thw'].to(args.device))
        grids = {k: spatial_grid(v, h, w) for k, v in captured.items()}
        # Concatenation changes the grid but preserves all values and their order.
        grids['premerger_grouped'] = captured['layer_27'].reshape(h // 2, w // 2, -1)
        grids['norm_grouped'] = captured['merger_norm'].reshape(h // 2, w // 2, -1)
        grids['visual_embedding'] = out.pooler_output.detach().float().cpu().reshape(h // 2, w // 2, -1)
        return grids

    started = time.perf_counter()
    baseline_image = figure_image()
    baseline_image.save(args.output / 'baseline.png')
    baseline = encode(baseline_image)
    repeat = encode(baseline_image)
    repeat_error = {k: float((baseline[k] - repeat[k]).abs().max()) for k in baseline}
    records, panels = [], []
    # Four fixed positions cover corner, edge, and interior, without cherry-picking.
    for gap in [1, 2, 4, 8]:
        for target in [0, 8, 21, 35]:
            im = figure_image(target, gap)
            stem = f'gap{gap}_target{target}'
            im.save(args.output / f'{stem}.png')
            features = encode(im)
            pixel_count = int(np.any(np.asarray(im) != np.asarray(baseline_image), axis=-1).sum())
            if pixel_count != 2 * gap:
                raise ValueError(f'Unexpected changed-pixel count: {pixel_count}')
            for name, grid in features.items():
                delta = grid - baseline[name]
                response = delta.square().mean(-1).sqrt()
                cells = cell_features(grid)
                median = cells.median(dim=0).values
                outlier = (cells - median).square().mean(-1).sqrt()
                # Ranks use strict wins; ties at the maximum are not successes.
                unique_top = int((outlier == outlier.max()).sum()) == 1
                delta_cells = cell_features(delta).square().sum(-1)
                total = float(delta_cells.sum())
                records.append(dict(gap=gap, target=target, changed_pixels=pixel_count, stage=name,
                    rank=1 + int((outlier > outlier[target]).sum()),
                    top1=bool(unique_top and int(outlier.argmax()) == target),
                    response_peak_cell=int(delta_cells.argmax()),
                    local_energy_fraction=float(delta_cells[target]) / total if total else 0.,
                    relative_change=float(delta.norm() / baseline[name].norm()),
                    outlier_scores=outlier.tolist(), response=response.tolist()))
                if gap == 2 and target == 21 and name in ['layer_3', 'layer_9', 'layer_18', 'layer_27', 'visual_embedding']:
                    panels.append((name, response, outlier.reshape(6, 6)))
            print(f'{stem}: {pixel_count} changed pixels', flush=True)
    summary = {name: {str(gap): sum(r['top1'] for r in records if r['stage'] == name and r['gap'] == gap)
                     for gap in [1, 2, 4, 8]} for name in baseline}
    metadata = dict(model=args.model, snapshot=model_path.name, transformers=transformers.__version__,
        torch=torch.__version__, device=torch.cuda.get_device_name(torch.device(args.device)), dtype='float32',
        config=json.loads((model_path / 'config.json').read_text()),
        processor_config=json.loads((model_path / 'preprocessor_config.json').read_text()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repeat_max_abs_error=repeat_error, top1_count_out_of_4=summary,
        elapsed_seconds=time.perf_counter() - started, records=records)
    (args.output / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    canvas = Image.new('RGB', (6 * 260, 600), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), '2x2-pixel gap; target row 4, column 4. Each heatmap independently scaled: blue=0, red=max.', fill='black')
    canvas.paste(figure_image(21, 2).resize((240, 240)), (10, 65))
    draw.text((10, 325), 'Top: paired feature response', fill='black')
    draw.text((10, 345), 'Bottom: single-image outlier score', fill='black')
    draw.text((10, 375), 'Known 6x6 figure boundaries.', fill='black')
    draw.text((10, 395), 'Paired response uses normal reference.', fill='black')
    draw.text((10, 415), 'Outlier score has no reference image.', fill='black')
    for j, (name, response, outlier) in enumerate(panels, start=1):
        x = j * 260 + 10
        draw.text((x, 40), name, fill='black')
        canvas.paste(heatmap(response), (x, 65))
        canvas.paste(heatmap(outlier), (x, 335))
    canvas.save(args.output / 'detail_maps.png')
    for handle in handles:
        handle.remove()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
