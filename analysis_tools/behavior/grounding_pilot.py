"""Controlled ordinal, salient-target, and fine-detail grounding pilot."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

from utils.project_paths import load_paths


def make_grid(n: int, uneven: bool, target: tuple[int, int], kind: str):
    """Keep figure pixels fixed while changing spacing; target indices are zero-based."""
    if n < 6 or kind not in ('normal', 'red', 'gap'):
        raise ValueError('Expected n >= 6 and a supported figure kind.')
    if any(i < 0 or i >= n for i in target):
        raise ValueError('Target outside grid.')
    widths = np.full(n, 64, dtype=int)
    heights = widths.copy()
    if uneven:
        # Equal canvas size, asymmetric spacing, and unchanged target ordinal.
        widths[:n // 2] = 48
        widths[n // 2:] = 80
        heights[:n // 2] = 80
        heights[n // 2:] = 48
    xs = np.cumsum(widths) - widths / 2
    ys = np.cumsum(heights) - heights / 2
    im = Image.new('RGB', (int(widths.sum()), int(heights.sum())), 'white')
    draw = ImageDraw.Draw(im)
    for r, cy in enumerate(ys):
        for c, cx in enumerate(xs):
            x, y = int(cx - 32), int(cy - 32)
            color = 'red' if kind == 'red' and (r, c) == target else 'black'
            draw.ellipse((x + 24, y + 8, x + 40, y + 24), outline=color, width=2)
            for line in [(32, 24, 32, 44), (16, 32, 48, 32), (32, 44, 20, 57), (32, 44, 44, 57)]:
                draw.line(tuple(v + (x if j % 2 == 0 else y) for j, v in enumerate(line)), fill=color, width=2)
            if kind == 'gap' and (r, c) == target:
                draw.rectangle((x + 39, y + 32, x + 46, y + 33), fill='white')
    # Center of the complete figure bounding box (including the unchanged legs).
    centers = [[float(x), float(y + .5)] for y in ys for x in xs]
    return im, centers


def score_point(response: str, centers: list[list[float]], target: int, size: tuple[int, int]):
    """Strict JSON parsing; invalid output remains an error in the denominator."""
    body = response.split('</think>')[-1].strip()
    body = re.sub(r'^```(?:json)?\s*|\s*```$', '', body)
    try:
        parsed = json.loads(body)
        if not isinstance(parsed, list) or len(parsed) != 1:
            raise ValueError('Expected exactly one point record.')
        point = parsed[0]['point_2d']
        valid = (isinstance(point, list) and len(point) == 2
                 and all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1000 for v in point))
    except (ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        return {'valid': False, 'identity_correct': False, 'precise': False}
    pixel = np.array(point) * np.array(size) / 1000
    distances = np.linalg.norm(np.array(centers) - pixel, axis=1)
    minimum = distances.min()
    unique = int(np.isclose(distances, minimum, atol=1e-6, rtol=0).sum()) == 1
    return {'valid': True, 'pixel': pixel.tolist(), 'identity_correct': bool(unique and distances.argmin() == target),
            'precise': bool(distances[target] <= 8), 'error_pixels': float(distances[target])}


def render_representation_panel(data: dict, output: Path) -> None:
    """Render normal and altered score maps with a shared scale within each layer."""
    stages = ['layer_3', 'layer_18', 'layer_27', 'visual_embedding']
    canvas = Image.new('RGB', (1100, 650), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), 'Single-image outlier scores: normal vs gap at row 4, column 4', fill='black')
    draw.text((15, 30), 'Same color scale within each column. Green outline: changed figure. Black outline: score maximum.', fill='black')
    for col, stage in enumerate(stages):
        normal = data['summary'][stage]['normal_scores']
        changed = next(r['scores'] for r in data['records'] if r['stage'] == stage and r['target'] == 21)
        peak = max(normal + changed)
        x = 15 + col * 275
        draw.text((x, 62), stage, fill='black')
        for row, (label, values) in enumerate([('All figures normal', normal), ('One figure has a gap', changed)]):
            y = 105 + row * 270
            draw.text((x, y-20), label, fill='black')
            for j, value in enumerate(values):
                a = value / peak
                box = (x+j%6*40, y+j//6*40, x+j%6*40+39, y+j//6*40+39)
                draw.rectangle(box, fill=(int(255*a), int(70*a), int(255*(1-a))))
            p = int(np.argmax(values))
            draw.rectangle((x+p%6*40+2,y+p//6*40+2,x+p%6*40+37,y+p//6*40+37), outline='black', width=3)
            if row:
                draw.rectangle((x+3*40+2,y+3*40+2,x+3*40+37,y+3*40+37), outline='#00dd55', width=3)
    canvas.save(output)


def render_behavior_panel(cases: list[dict], scored: list[dict], output: Path) -> None:
    """Show a fixed selection of successful and failed points without changing scores."""
    ids = ['n6_uneven0_r5_c6_ordinal', 'n6_uneven0_r5_c6_odd',
           'n20_uneven0_r5_c6_ordinal', 'n20_uneven0_r5_c6_red']
    canvas = Image.new('RGB', (1480, 470), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), 'Grounding pilot: green circle = true center; orange cross = model point', fill='black')
    for col, case_id in enumerate(ids):
        case = next(c for c in cases if c['id'] == case_id)
        result = next(s for s in scored if s['id'] == case_id)
        with Image.open(case['image']) as im:
            thumb = im.convert('RGB').resize((350,350))
        x, y = 10+col*370, 65
        canvas.paste(thumb, (x,y))
        draw.text((x,40), f"{case['n']}x{case['n']} / {case['task']} / target (5,6)", fill='black')
        scale = 350 / case['size'][0]
        tx, ty = case['centers'][case['target']]
        tx, ty = x+tx*scale, y+ty*scale
        draw.ellipse((tx-8,ty-8,tx+8,ty+8), outline='#00a050', width=3)
        if result['valid']:
            px, py = result['pixel']
            px, py = x+px*scale, y+py*scale
            draw.line((px-7,py-7,px+7,py+7), fill='#ff7200', width=3)
            draw.line((px-7,py+7,px+7,py-7), fill='#ff7200', width=3)
            draw.text((x,430), f"Center error: {result['error_pixels']:.1f} original pixels", fill='black')
    canvas.save(output)


def behavior(args):
    """Reuse the repository inference runner and preserve every raw response."""
    from eva.inference import main as infer
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    suffix = (' Return only a JSON list with one item: [{"point_2d": [x, y]}]. Do not explain. Coordinates are normalized to 0-1000: '
              'x increases from left to right, y from top to bottom. Give the center of the figure bounding box.')
    for n in (6, 20):
        for uneven in (False, True):
            for target in ((1, 2), (4, 5)):
                for task, kind in [('ordinal', 'normal'), ('red', 'red'), ('odd', 'gap'), ('known_odd', 'gap')]:
                    stem = f'n{n}_uneven{int(uneven)}_r{target[0]+1}_c{target[1]+1}_{task}'
                    im, centers = make_grid(n, uneven, target, kind)
                    path = args.output / f'{stem}.png'
                    im.save(path)
                    query = (f'Find the person in row {target[0]+1}, column {target[1]+1}. '
                             'Count rows from top to bottom and columns from left to right, starting at 1.')
                    if task == 'red':
                        query = 'Find the red person.'
                    elif task == 'odd':
                        query = 'Find the one person with a gap in its right arm (on the right side of the image).'
                    cases.append(dict(id=stem, n=n, uneven=uneven, task=task, image=str(path),
                                      prompt=query + suffix, centers=centers, target=target[0]*n+target[1],
                                      size=im.size, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (args.output / 'experiment.json').write_text(json.dumps(dict(
        schema='single_point_2d_v2', code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_new_tokens=args.max_new_tokens, seed=0, precise_threshold_pixels=8), indent=2) + '\n')
    prediction = args.output / 'predictions.jsonl'
    sys.argv = ['grounding_pilot', '--model', args.model, '--output', str(prediction),
                '--gpu-ids', args.gpu_ids, '--batchsize', '1', '--max-new-tokens', str(args.max_new_tokens),
                '--thinking', 'off', '--attn-implementation', 'sdpa',
                '--image-path', *[c['image'] for c in cases], '--prompt', *[c['prompt'] for c in cases]]
    infer()
    records = [json.loads(s) for s in prediction.read_text().splitlines()]
    if len(records) != len(cases):
        raise ValueError('Incomplete predictions.')
    groups = defaultdict(list)
    scored = []
    for case, record in zip(cases, records, strict=True):
        if record['image_path'] != case['image']:
            raise ValueError('Prediction order mismatch.')
        result = score_point(record['response'], case['centers'], case['target'], tuple(case['size']))
        if record['finish_reason'] != 'eos':
            result.update(valid=False, identity_correct=False, precise=False)
        scored.append(dict(id=case['id'], response=record['response'], **result))
        groups[f"n{case['n']}_{case['task']}_uneven{int(case['uneven'])}"].append(result)
    summary = {k: dict(total=len(v), **{metric: sum(x[metric] for x in v)
                                      for metric in ('valid', 'identity_correct', 'precise')}) for k, v in groups.items()}
    (args.output / 'scores.json').write_text(json.dumps(dict(summary=summary, records=scored), indent=2) + '\n')
    render_behavior_panel(cases, scored, args.output / 'point_examples.png')
    print(json.dumps(summary, indent=2))


def representation(args):
    """Compare single-image anomaly scores to the all-normal spatial score map."""
    import torch
    import transformers
    from transformers import AutoImageProcessor
    from analysis_tools.representation.visual_detail import load_vision, figure_image, spatial_grid, cell_features
    from eva.inference import resolve_model
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    model_path = resolve_model(args.model, load_paths()['HF_HUB_CACHE'])
    model = load_vision(model_path, 'cuda:0')
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    captured = {}
    def hook(name):
        def capture(module, inputs, output):
            captured[name] = output.detach()
        return capture
    handles = [model.patch_embed.register_forward_hook(hook('patch_embed'))]
    handles += [model.blocks[i-1].register_forward_hook(hook(f'layer_{i}')) for i in (3, 9, 18, 27)]
    @torch.inference_mode()
    def encode(im):
        batch = processor(images=im, return_tensors='pt').to('cuda:0')
        if batch.image_grid_thw.tolist() != [[1, 24, 24]]:
            raise ValueError('Unexpected processor resize.')
        out = model(batch.pixel_values.float(), grid_thw=batch.image_grid_thw)
        cells = {k: cell_features(spatial_grid(v, 24, 24)) for k, v in captured.items()}
        cells['visual_embedding'] = cell_features(out.pooler_output.reshape(12, 12, -1))
        return cells
    def scores(x):
        return (x - x.median(dim=0).values).square().mean(-1).sqrt()
    normal = encode(figure_image())
    base = {k: scores(x) for k, x in normal.items()}
    records = []
    for target in range(36):
        changed = encode(figure_image(target, 8))
        for stage, x in changed.items():
            s, b = scores(x), base[stage]
            energy = (x-normal[stage]).square().sum(-1)
            correlation = None if float(b.std()) == 0 or float(s.std()) == 0 else float(torch.corrcoef(torch.stack([b,s]))[0,1])
            pred = int(s.argmax())
            records.append(dict(target=target, stage=stage, prediction=pred,
                                top1=bool(pred == target and (s == s.max()).sum() == 1),
                                normal_peak=int(b.argmax()), same_normal_peak=bool(pred == int(b.argmax())),
                                correlation=correlation, paired_peak=int(energy.argmax()),
                                scores=s.cpu().tolist()))
    summary = {k: dict(top1=sum(r['top1'] for r in records if r['stage']==k),
                       same_normal_peak=sum(r['same_normal_peak'] for r in records if r['stage']==k),
                       paired_top1=sum(r['paired_peak']==r['target'] for r in records if r['stage']==k),
                       mean_correlation=float(np.mean([r['correlation'] for r in records if r['stage']==k and r['correlation'] is not None]))
                       if float(base[k].std()) else None,
                       normal_scores=base[k].cpu().tolist()) for k in normal}
    data = dict(model=args.model, snapshot=model_path.name, torch=torch.__version__,
                transformers=transformers.__version__, dtype='float32', gap=8, targets=36,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), summary=summary, records=records)
    (args.output / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    render_representation_panel(data, args.output / 'normal_control.png')
    for handle in handles:
        handle.remove()
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['behavior', 'representation'], required=True)
    parser.add_argument('--model', default='Qwen3.5-9B')
    parser.add_argument('--gpu-ids', default='1')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be positive')
    if args.mode == 'behavior':
        behavior(args)
    else:
        representation(args)


if __name__ == '__main__':
    main()
