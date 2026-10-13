"""Uncalibrated raw logit lens on Qwen3.5 visual residual positions."""
import argparse
from collections import Counter
import hashlib
import html
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis_tools.grounding_pilot import make_grid
from utils.project_paths import load_paths


def norm_groups(norms: list[float]) -> tuple[list[int], list[int]]:
    """Partition by fixed top-decile rank; this does not identify registers."""
    values = np.asarray(norms)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Expected at least two finite token norms.')
    order = np.argsort(values, kind='stable')
    count = max(1, int(np.ceil(len(order) * .1)))
    return order[-count:].tolist(), order[:-count].tolist()


def collect_cases(paths: dict) -> list[tuple[str, Image.Image, dict]]:
    """Fix cases before observing any activations; natural images are deduplicated."""
    import pyarrow.parquet as pq
    cases = [(f'people_{kind}', make_grid(6, False, (3, 3), kind)[0], {'kind': kind})
             for kind in ('normal', 'gap')]
    for shape in ('circles', 'triangles'):
        for altered in (False, True):
            im = Image.new('RGB', (384, 384), 'white')
            draw = ImageDraw.Draw(im)
            for r in range(6):
                for c in range(6):
                    x, y = c * 64, r * 64
                    if shape == 'circles':
                        draw.ellipse((x+14, y+14, x+50, y+50), outline='black', width=3)
                    else:
                        draw.polygon([(x+32,y+12),(x+12,y+50),(x+52,y+50)], outline='black', width=3)
                    if altered and (r, c) == (3, 3):
                        draw.rectangle((x+29,y+47,x+36,y+52), fill='white')
            cases.append((f'{shape}_{"gap" if altered else "normal"}', im, {'shape': shape, 'altered': altered}))
    root = Path(paths['DATA_DIR']) / 'BabyVision'
    wanted = {507, 445, 638}
    for line in (root / 'train.jsonl').read_text().splitlines():
        record = json.loads(line)
        if record['taskId'] in wanted:
            cases.append((f'babyvision_{record["taskId"]}', Image.open(root / record['image']).convert('RGB'),
                          {'task_id': record['taskId'], 'source': record['image']}))
            wanted.remove(record['taskId'])
    if wanted:
        raise ValueError(f'Missing BabyVision cases: {wanted}')
    refroot = Path(os.environ['REFCOCO_DIR'])
    seen = set()
    for batch in pq.ParquetFile(refroot / 'data/val-00000-of-00004.parquet').iter_batches(batch_size=16):
        for row in batch.to_pylist():
            digest = hashlib.sha256(row['image']['bytes']).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            cases.append((f'refcoco_{len(seen)}', Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB'),
                          {'file_name': row['file_name'], 'question_id': row['question_id'],
                           'bbox_xywh': row['bbox'], 'referring_expressions': row['answer'], 'source_bytes_sha256': digest}))
            if len(seen) == 3:
                return cases
    raise ValueError('Fewer than three unique RefCOCO images.')


def decode_states(states, norm, head, tokenizer, chunk_size: int = 32) -> dict:
    """Apply the learned final norm once, with bounded vocabulary allocations."""
    import torch
    records = []
    for start in range(0, len(states), chunk_size):
        h = states[start:start+chunk_size].to(head.weight.device, head.weight.dtype)
        logits = head(norm(h)).float()
        logp = logits.log_softmax(-1)
        probs, ids = logp.topk(3, dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        for j in range(len(h)):
            records.append({'norm': float(h[j].float().norm()), 'entropy': float(entropy[j]),
                            'top_ids': ids[j].tolist(), 'top_tokens': [tokenizer.decode([i]) for i in ids[j].tolist()],
                            'top_probs': probs[j].exp().tolist()})
    high, ordinary = norm_groups([r['norm'] for r in records])
    summary = {}
    for label, indices in [('top_norm_decile', high), ('other_tokens', ordinary)]:
        rows = [records[i] for i in indices]
        words = [r['top_tokens'][0] for r in rows]
        summary[label] = {'n': len(rows), 'mean_norm': float(np.mean([r['norm'] for r in rows])),
                          'mean_entropy': float(np.mean([r['entropy'] for r in rows])),
                          'mean_top1_probability': float(np.mean([r['top_probs'][0] for r in rows])),
                          'nonalphanumeric_fraction': float(np.mean([not any(c.isalnum() for c in s) for s in words])),
                          'top_words': Counter(words).most_common(8)}
    return {'tokens': records, 'groups': summary,
            'max_norm_over_median': float(max(r['norm'] for r in records) / np.median([r['norm'] for r in records]))}


def visible_token(token: str) -> str:
    """Render Unicode directly while making whitespace explicit."""
    return token.replace("\r", "[CR]").replace("\n", "[LF]").replace("\t", "[TAB]").replace(" ", "·") or "∅"


def render_case(case: dict, directory: Path, font_path: str | None = None) -> None:
    """Draw all three ranks with a CJK font and exact UTF-8 HTML labels."""
    font_path = font_path or os.environ.get('TOKEN_MAP_FONT')
    if not font_path or not Path(font_path).is_file():
        raise ValueError('Set TOKEN_MAP_FONT to a font supporting Chinese, e.g. Noto Sans CJK.')
    font = ImageFont.truetype(font_path, 14)
    small = ImageFont.truetype(font_path, 11)
    h, w = case['grid']
    pages = ['<!doctype html><meta charset="utf-8"><title>Raw logit lens</title>',
             '<style>td{min-width:88px;height:44px;border:1px solid #bbb;font:14px "Noto Sans CJK SC",sans-serif}table{border-collapse:collapse}small{display:block;color:#666}</style>',
             '<h1>Raw logit lens: 未校准词表读出</h1><p>红边框：范数最高的10%。·为空格，[LF]为换行，[TAB]为制表符。每格下方为该词概率；悬停查看完整记录。</p>',
             '<img width="480" src="image.png">']
    for layer, result in case['layers'].items():
        high = set(norm_groups([r['norm'] for r in result['tokens']])[0])
        for rank in range(3):
            pages.append(f'<h2>{layer} · top-{rank+1}</h2><table>')
            canvas = Image.new('RGB', (max(w*116,480), h*53+56), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((4,4), f'{layer} · top-{rank+1} · 红框：高范数 · 每格下方：概率', fill='black', font=font)
            for r in range(h):
                pages.append('<tr>')
                for c in range(w):
                    i = r*w+c
                    token = result['tokens'][i]
                    color = '#bb2222' if i in high else '#cccccc'
                    title = html.escape(json.dumps({'row': r, 'column': c, **token}, ensure_ascii=False), quote=True)
                    word = visible_token(token['top_tokens'][rank])
                    probability = token['top_probs'][rank]
                    pages.append(f'<td style="border:2px solid {color}" title="{title}">{html.escape(word)}<small>{probability:.4f}</small></td>')
                    x, y = c*116, r*53+48
                    draw.rectangle((x,y,x+115,y+52), outline=color, width=2)
                    label = word
                    while draw.textlength(label, font=font)>106 and len(label)>1:
                        label = label[:-2]+'…'
                    draw.text((x+4,y+4), label, fill='black', font=font)
                    draw.text((x+4,y+28), f'{probability:.4f}', fill='#555555', font=small)
                pages.append('</tr>')
            pages.append('</table>')
            canvas.save(directory / f'{layer}_top{rank+1}.png')
    (directory / 'index.html').write_text('\n'.join(pages), encoding='utf-8')


def main() -> None:
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from eva.inference import resolve_model
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', default='Qwen3.5-9B')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 24, 32])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    paths = load_paths()
    cases = collect_cases(paths)
    checkpoint = resolve_model(args.model, paths['HF_HUB_CACHE'])
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, local_files_only=True, dtype=torch.bfloat16,
                                                     device_map='cuda:0', attn_implementation='sdpa').eval()
    language = model.model.language_model
    selected = tuple(args.layers)
    if len(set(selected)) != len(selected) or any(i < 0 or i > len(language.layers) for i in selected):
        raise ValueError("Layers must be unique and within the model depth.")
    captured = {}
    mask = None
    def capture(layer):
        def hook(module, inputs, output):
            value = inputs[0] if layer == 0 else output
            if isinstance(value, tuple):
                value = value[0]
            captured[layer] = value[0, mask].detach().cpu()
        return hook
    handles = [language.layers[0].register_forward_hook(capture(0))]
    handles += [language.layers[i-1].register_forward_hook(capture(i)) for i in selected if i]
    summary = []
    with torch.inference_mode():
        for name, image, meta in cases:
            directory = args.output / name
            directory.mkdir()
            image.save(directory / 'image.png')
            chat = [{'role': 'user', 'content': [{'type':'image'}, {'type':'text','text':'Describe the image.'}]}]
            text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            batch = processor(text=[text], images=[image], return_tensors='pt').to('cuda:0')
            mask = batch.input_ids[0] == model.config.image_token_id
            t, gh, gw = batch.image_grid_thw[0].tolist()
            if t != 1 or int(mask.sum()) != gh*gw//4:
                raise ValueError('Unexpected visual token layout.')
            captured.clear()
            model.model(**batch, use_cache=False)
            record = {'id': name, 'source': meta, 'original_size': image.size, 'grid': [gh//2,gw//2],
                      'image_sha256': hashlib.sha256((directory/'image.png').read_bytes()).hexdigest(), 'layers': {}}
            for layer in selected:
                record['layers'][f'layer_{layer}'] = decode_states(captured[layer], language.norm, model.lm_head, processor.tokenizer)
            (directory/'tokens.json').write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
            render_case(record, directory)
            summary.append({**{k:v for k,v in record.items() if k != 'layers'},
                            'layers': {k:{kk:vv for kk,vv in v.items() if kk != 'tokens'} for k,v in record['layers'].items()}})
            print(f'Finished {name}: {gh//2} x {gw//2} visual tokens', flush=True)
    for handle in handles:
        handle.remove()
    (args.output/'summary.json').write_text(json.dumps({'model':args.model, 'checkpoint': checkpoint.name,
        'torch':torch.__version__, 'transformers':transformers.__version__, 'thinking':'off',
        'prompt':'Describe the image.', 'dtype':'bfloat16', 'layers':selected,
        'font_sha256':hashlib.sha256(Path(os.environ['TOKEN_MAP_FONT']).read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases':summary}, indent=2, ensure_ascii=False)+'\n')
    links = ''.join(f'<li><a href="{r["id"]}/index.html">{r["id"]}</a></li>' for r in summary)
    (args.output/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>Uncalibrated raw logit lens</h1><ul>{links}</ul>')


if __name__ == '__main__':
    main()
