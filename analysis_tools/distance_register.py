"""Distance, token norm, and controlled-edit probes of frozen Qwen3.5 vision features."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from analysis_tools.visual_detail import cell_features, figure_image, heatmap, load_vision, spatial_grid
from eva.inference import resolve_model
from utils.project_paths import load_paths


def synthetic(family: str, shift: int, target: int | None = None) -> Image.Image:
    """Preserve repeated cell pixels, including their subpatch phase, within each image."""
    if family == 'person':
        image = figure_image(target, 8 if target is not None else 0)
    else:
        image = Image.new('RGB', (384, 384), 'white')
        draw = ImageDraw.Draw(image)
        for i in range(36):
            x, y = i % 6 * 64, i // 6 * 64
            if family == 'ring':
                draw.ellipse((x+12, y+12, x+50, y+50), outline='black', width=3)
            elif family == 'square':
                draw.rectangle((x+12, y+12, x+50, y+50), outline='black', width=3)
            else:
                raise ValueError(f'Unknown family: {family}')
            if i == target:
                draw.rectangle((x+28, y+12, x+35, y+15), fill='white')
    if shift:
        shifted = Image.new('RGB', image.size, 'white')
        shifted.paste(image, (shift, shift))
        image = shifted
    return image


def paired_maps(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compare corresponding spatial tokens; cosine measures direction changes."""
    delta = a-b
    return {'rms': delta.square().mean(-1).sqrt(), 'l2': delta.norm(dim=-1),
            'cosine': (1-F.cosine_similarity(a, b, dim=-1)).clamp_min(0)}


def single_scores(grid: torch.Tensor) -> dict[str, torch.Tensor]:
    """All transforms use only this image; suppression replaces tokens before readout."""
    norms = grid.norm(dim=-1)
    median_norm = norms.median().clamp_min(1e-12)
    threshold = torch.quantile(norms, .95)
    count = max(1, math.ceil(norms.numel()*.05))
    high = norms.flatten().topk(count).indices
    replacement = grid.reshape(-1, grid.shape[-1]).median(0).values
    results = {}
    variants = {'rms_median': grid,
                'unit_token_rms': F.normalize(grid, dim=-1),
                'cap95_rms': grid*(threshold/norms.clamp_min(1e-12)).clamp_max(1)[..., None],
                'cap_median_rms': grid*(median_norm/norms.clamp_min(1e-12)).clamp_max(1)[..., None]}
    for label, ids in [('replace_high5_rms', high)] + [
            (f'replace_random5_seed{seed}', torch.randperm(norms.numel(), generator=torch.Generator().manual_seed(seed))[:count])
            for seed in range(5)]:
        modified = grid.clone().reshape(-1, grid.shape[-1])
        modified[ids] = replacement
        variants[label] = modified.reshape_as(grid)
    for name, features in variants.items():
        cells = cell_features(features)
        results[name] = (cells-cells.median(0).values).square().mean(-1).sqrt()
    cells = cell_features(grid)
    results['cosine_median'] = (1-F.cosine_similarity(cells, cells.median(0).values[None], dim=-1)).clamp_min(0)
    # Mean distance to other cells avoids relying on a coordinate-wise median direction.
    unit = F.normalize(cells, dim=-1)
    results['cosine_pairwise_mean'] = (1-unit@unit.T).clamp_min(0).sum(-1)/35
    return results


def norm_record(grid: torch.Tensor) -> dict:
    values = grid.norm(dim=-1)
    median = float(values.median())
    q = torch.quantile(values, torch.tensor([0., .5, .9, .95, .99, 1.]))
    return {'shape': list(grid.shape), 'quantiles': q.tolist(), 'max_over_median': float(q[-1])/max(median, 1e-12),
            'fraction_above_3median': float((values > 3*median).float().mean()),
            'fraction_above_10median': float((values > 10*median).float().mean()),
            'norm_map': values.tolist(), 'high5_flat_indices': values.flatten().topk(max(1, math.ceil(values.numel()*.05))).indices.tolist()}


def unique_top(scores: torch.Tensor) -> int | None:
    """Ties, including all-zero controls, do not count as a selected instance."""
    return int(scores.argmax()) if int((scores == scores.max()).sum()) == 1 else None


def checkpoint_metadata(model_name: str, model_path: Path) -> dict:
    """Bind recovery to model identity, preprocessing, numerical settings, and code."""
    return {'schema_version': 2, 'model': model_name, 'snapshot': model_path.name,
            'dtype': 'float32', 'seed': 0, 'allow_tf32': False,
            'families': ['person', 'ring', 'square'], 'shifts': [0, 4],
            'targets': list(range(36)),
            'stages': ['patch_embed', 'layer_3', 'layer_9', 'layer_18', 'layer_27', 'visual_embedding'],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'vision_helpers_sha256': hashlib.sha256(Path(__file__).with_name('visual_detail.py').read_bytes()).hexdigest(),
            'config_sha256': hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
            'processor_sha256': hashlib.sha256((model_path/'preprocessor_config.json').read_bytes()).hexdigest()}


def validate_checkpoint(checkpoint: dict, expected: dict) -> None:
    """Reject legacy, incompatible, incomplete, or duplicate synthetic results."""
    if checkpoint.get('metadata') != expected:
        raise ValueError('Synthetic checkpoint metadata is missing or incompatible with this run.')
    normals, records = checkpoint.get('normals', []), checkpoint.get('records', [])
    cases = [f'{family}_shift{shift}' for family in expected['families'] for shift in expected['shifts']]
    wanted_normals = {(case, stage) for case in cases for stage in expected['stages']}
    wanted_records = {(case, stage, target) for case, stage in wanted_normals for target in expected['targets']}
    normal_keys = [(r['case'], r['stage']) for r in normals]
    record_keys = [(r['case'], r['stage'], r['target']) for r in records]
    if (len(normal_keys) != len(wanted_normals) or set(normal_keys) != wanted_normals
            or len(record_keys) != len(wanted_records) or set(record_keys) != wanted_records):
        raise ValueError('Synthetic checkpoint is incomplete or contains duplicate sample keys.')


def external_peak_record(values: torch.Tensor, target: tuple[int, int]) -> dict:
    """Retain argmax for inspection but require uniqueness when scoring new runs."""
    flat = values.flatten()
    top = unique_top(flat)
    return {'peak_token': [int(i) for i in np.unravel_index(int(flat.argmax()), values.shape)],
            'unique_peak': top is not None,
            'correct': top is not None and top == target[0]*values.shape[1]+target[1],
            'local_squared_response_fraction': float(values[target].square()/values.square().sum().clamp_min(1e-20))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refcoco-dir', type=Path, required=True)
    parser.add_argument('--model', default='Qwen3.5-9B')
    parser.add_argument('--resume-synthetic', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume_synthetic)
    paths = load_paths()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_path = resolve_model(args.model, paths['HF_HUB_CACHE'])
    metadata = checkpoint_metadata(args.model, model_path)
    checkpoint = None
    if args.resume_synthetic:
        checkpoint = json.loads((args.output/'synthetic_checkpoint.json').read_text())
        validate_checkpoint(checkpoint, metadata)
    model = load_vision(model_path, 'cuda:0')
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    captured = {}
    def hook(name):
        def capture(module, inputs, output):
            captured[name] = output.detach().float().cpu()
        return capture
    handles = [model.patch_embed.register_forward_hook(hook('patch_embed'))]
    for i in [3, 9, 18, 27]:
        handles.append(model.blocks[i-1].register_forward_hook(hook(f'layer_{i}')))

    @torch.inference_mode()
    def encode(image):
        batch = processor(images=image, return_tensors='pt')
        t, h, w = batch['image_grid_thw'][0].tolist()
        if t != 1:
            raise ValueError('Only still images are supported.')
        output = model(batch['pixel_values'].to('cuda:0', torch.float32), grid_thw=batch['image_grid_thw'].to('cuda:0'))
        grids = {k: spatial_grid(v, h, w) for k, v in captured.items()}
        grids['visual_embedding'] = output.pooler_output.detach().float().cpu().reshape(h//2, w//2, -1)
        return grids

    records, normals, examples, external = [], [], {}, []
    started = time.time()
    if not args.resume_synthetic:
        for family in ['person', 'ring', 'square']:
            for shift in [0, 4]:
                case = f'{family}_shift{shift}'
                original = synthetic(family, shift)
                original.save(args.output/f'{case}_normal.png')
                baseline = encode(original)
                normal_scores = {stage: single_scores(grid) for stage, grid in baseline.items()}
                for stage, grid in baseline.items():
                    scores = normal_scores[stage]
                    norm_cells = cell_features(grid.norm(dim=-1, keepdim=True)).mean(-1)
                    correlation = float(np.corrcoef(norm_cells.numpy(), scores['rms_median'].numpy())[0, 1]) if float(scores['rms_median'].std()) > 1e-10 else None
                    normals.append({'case': case, 'stage': stage, 'norm': norm_record(grid), 'norm_score_pearson': correlation,
                                    'scores': {k: v.tolist() for k,v in scores.items()},
                                    'peaks': {k: unique_top(v) for k,v in scores.items()}})
                for target in range(36):
                    image = synthetic(family, shift, target)
                    grids = encode(image)
                    for stage, grid in grids.items():
                        scores = single_scores(grid)
                        paired = paired_maps(grid, baseline[stage])
                        pair_cells = {k: cell_features(v[..., None]).square().mean(-1).sqrt() for k,v in paired.items()}
                        records.append({'case': case, 'family': family, 'shift': shift, 'target': target, 'stage': stage,
                                        'single': {k: {'top': unique_top(v), 'correct': unique_top(v)==target,
                                                       'normal_peak_same': unique_top(v) is not None and unique_top(v)==unique_top(normal_scores[stage][k])}
                                                   for k,v in scores.items()},
                                        'paired': {k: {'top': unique_top(v), 'correct': unique_top(v)==target} for k,v in pair_cells.items()}})
                        if shift == 0 and target == 21 and stage == 'visual_embedding':
                            examples[family] = {'image': image, 'normal_norm': baseline[stage].norm(dim=-1),
                                                'normal_score': normal_scores[stage]['rms_median'].reshape(6,6),
                                                'rms': scores['rms_median'].reshape(6,6),
                                                'cosine': scores['cosine_median'].reshape(6,6),
                                                'unit': scores['unit_token_rms'].reshape(6,6), 'paired': paired['cosine']}
                    print(f'{case} target {target}', flush=True)
                (args.output/'synthetic_checkpoint.json').write_text(json.dumps({'metadata': metadata, 'normals': normals, 'records': records}))

    else:
        normals, records = checkpoint['normals'], checkpoint['records']
        for family in ['person', 'ring', 'square']:
            image = synthetic(family, 0, 21)
            normal = encode(synthetic(family,0))['visual_embedding']
            grid = encode(image)['visual_embedding']
            scores = single_scores(grid)
            examples[family] = {'image':image, 'normal_norm':normal.norm(dim=-1),
                                'normal_score':single_scores(normal)['rms_median'].reshape(6,6),
                                'rms':scores['rms_median'].reshape(6,6),'cosine':scores['cosine_median'].reshape(6,6),
                                'unit':scores['unit_token_rms'].reshape(6,6),'paired':paired_maps(grid,normal)['cosine']}

    # Dataset images are descriptive controls: no natural object labels are inferred.
    sources = []
    for line in (Path(paths['DATA_DIR'])/'BabyVision/train.jsonl').read_text().splitlines():
        item = json.loads(line)
        if item['taskId'] in [507,445,638]:
            sources.append((f"babyvision_{item['taskId']}", Image.open(Path(paths['DATA_DIR'])/'BabyVision'/item['image']).convert('RGB'), {'task_id': item['taskId']}))
    if len(sources) != 3:
        raise ValueError('Missing requested BabyVision samples.')
    import pyarrow.parquet as pq
    seen = set()
    for batch in pq.ParquetFile(args.refcoco_dir/'data/val-00000-of-00004.parquet').iter_batches(batch_size=16):
        for item in batch.to_pylist():
            content = item['image']['bytes']
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            sources.append((f'refcoco_{len(seen)}', Image.open(io.BytesIO(content)).convert('RGB'), {'file_name': item['file_name'], 'image_sha256': digest}))
            if len(seen) == 6:
                break
        if len(seen) == 6:
            break
    if len(seen) != 6:
        raise ValueError('Need six distinct RefCOCO images.')
    for name, original, metadata in sources:
        # Resize once to a reproducible processor-compatible grid before editing.
        scale = min(1., 640/max(original.size))
        size = tuple(max(32, round(v*scale/32)*32) for v in original.size)
        image = original.resize(size, Image.Resampling.BICUBIC)
        image.save(args.output/f'{name}.png')
        baseline = encode(image)
        for stage, grid in baseline.items():
            external.append({'case':name, 'stage':stage, 'kind':'norm', 'source_size':list(original.size), 'analysis_size':size,
                             'metadata':metadata, 'norm':norm_record(grid)})
        for edit, (fx,fy) in enumerate([(0.25,0.25),(.5,.5),(.75,.75)]):
            x,y = int(size[0]*fx)//32*32, int(size[1]*fy)//32*32
            altered = np.array(image)
            altered[y:y+8,x:x+8] = 255-altered[y:y+8,x:x+8]
            changed = Image.fromarray(altered)
            changed.save(args.output/f'{name}_edit{edit}.png')
            grids = encode(changed)
            for stage, grid in grids.items():
                h,w,_ = grid.shape
                target = (y*h//size[1], x*w//size[0])
                maps = paired_maps(grid,baseline[stage])
                external.append({'case':name,'stage':stage,'kind':'edit','edit':edit,'box':[x,y,x+8,y+8], 'target_token':target,
                                 'paired': {k: external_peak_record(v, target) for k,v in maps.items()}})
        print(f'External {name}',flush=True)
    result={'model':args.model,'snapshot':model_path.name,'dtype':'float32','seed':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'elapsed_seconds':time.time()-started,'normals':normals,'records':records,'external':external}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    canvas=Image.new('RGB',(7*210,3*245),'white')
    draw=ImageDraw.Draw(canvas)
    columns=['image','normal_norm','normal_score','rms','cosine','unit','paired']
    for row,(family,data) in enumerate(examples.items()):
        for col,key in enumerate(columns):
            draw.text((col*210+5,row*245+5),f'{family}: {key}',fill='black')
            pic=data[key].resize((200,200)) if key=='image' else heatmap(data[key],200)
            canvas.paste(pic,(col*210+5,row*245+30))
    canvas.save(args.output/'metric_comparison.png')
    external_canvas = Image.new('RGB', (9*180, 3*200), 'white')
    external_draw = ImageDraw.Draw(external_canvas)
    for col, (name, _, _) in enumerate(sources):
        external_draw.text((col*180+4,4),name,fill='black')
        external_canvas.paste(Image.open(args.output/f'{name}.png').resize((170,170)),(col*180+4,24))
        for row, stage in enumerate(['layer_18','visual_embedding'],start=1):
            item = next(r for r in external if r['case']==name and r['kind']=='norm' and r['stage']==stage)
            external_draw.text((col*180+4,row*200+4),f"{stage}: {item['norm']['max_over_median']:.1f}x",fill='black')
            external_canvas.paste(heatmap(torch.tensor(item['norm']['norm_map']),170),(col*180+4,row*200+24))
    external_canvas.save(args.output/'external_norms.png')
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()
