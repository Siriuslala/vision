"""Render existing grounding responses without rerunning or changing their scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype('DejaVuSans.ttf', size)


def panel(image: Image.Image, title: str, lines: list[str], box=None, point=None, predicted_box=None) -> Image.Image:
    width, height = image.size
    pic = image.convert('RGB').resize((384, 384))
    draw = ImageDraw.Draw(pic)
    scale = np.array([384 / width, 384 / height])
    if box is not None:
        bounds = np.array(box) * np.tile(scale, 2)
        draw.rectangle(tuple(bounds), outline='#008c45', width=3)
    if predicted_box is not None:
        bounds = np.array(predicted_box) * np.tile(scale, 2)
        draw.rectangle(tuple(bounds), outline='#b000bb', width=3)
    if point is not None:
        x, y = np.array(point) * scale
        draw.ellipse((x-6, y-6, x+6, y+6), fill='#0071bc', outline='white', width=2)
    canvas = Image.new('RGB', (420, 470 + 22 * len(lines)), 'white')
    canvas.paste(pic, (18, 45))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), title, font=font(19), fill='black')
    for i, line in enumerate(lines):
        draw.text((18, 440 + i * 22), line, font=font(15), fill='black')
    return canvas


def join(panels: list[Image.Image], path: Path) -> None:
    image = Image.new('RGB', (sum(x.width for x in panels), max(x.height for x in panels)), 'white')
    x = 0
    for item in panels:
        image.paste(item, (x, 0))
        x += item.width
    image.save(path)


def render(source: Path, output: Path, document_figures: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((source / 'cases.json').read_text())
    records = [json.loads(line) for line in (source / 'predictions.jsonl').read_text().splitlines()]
    if len(cases) != len(records):
        raise ValueError('Case and response counts disagree.')
    responses = dict(zip((c['id'] for c in cases), records, strict=True))
    scores = {r['id']: r for r in json.loads((source / 'scores_geometry.json').read_text())['records']}
    panels = []
    for minority, title in [('gap', '35 intact + 1 gap'), ('intact', '35 gaps + 1 intact'), ('red', '35 black + 1 red')]:
        case = next(c for c in cases if c.get('minority') == minority and c.get('target') == 21 and c['format'] == 'point' and c['task'] == 'attribute')
        panels.append(panel(Image.open(case['image']), title, ['Target: row 4, column 4', 'Green box: annotation only'], box=case['bbox']))
    join(panels, output / 'construction.png')
    panels = []
    for target in (8, 21, 28):
        case = next(c for c in cases if c.get('minority') == 'red' and c.get('target') == target and c['format'] == 'point')
        row_case = next(c for c in cases if c.get('minority') == 'red' and c.get('target') == target and c['format'] == 'row_column')
        score = scores[case['id']]
        coord = np.round(np.array(score['pixel']) / np.array(case['size']) * 1000).astype(int)
        row_answer = json.loads(responses[row_case['id']]['response'])
        panels.append(panel(Image.open(case['image']), f'Target row {target//6+1}, column {target%6+1}',
                            [f'Point answer: ({coord[0]}, {coord[1]})', f'Row/column answer: ({row_answer["row"]}, {row_answer["column"]})',
                             'Green: true box; blue: predicted point'], box=case['bbox'], point=score['pixel']))
    join(panels, output / 'red_point_answers.png')
    panels = []
    for case in [c for c in cases if c['source'] == 'refcoco' and c['format'] == 'bbox'][:3]:
        score = scores[case['id']]
        expr = case['referring_expression']
        panels.append(panel(Image.open(case['image']), f'RefCOCO ID {case["question_id"]}',
                            [expr[:46], f'IoU = {score["iou"]:.3f}', 'Green: true box; purple: predicted box'],
                            box=case['bbox'], predicted_box=score['pixel']))
    join(panels, output / 'natural_bbox_answers.png')
    provenance = {'kind': 'visualization_of_existing_run', 'source_directory': str(source),
                  'source_files': {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                   for name in ['cases.json', 'predictions.jsonl', 'scores.json', 'scores_geometry.json']},
                  'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'note': 'No generation or scoring was performed. Boxes shown only after inference.'}
    (output/'manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    document_figures.mkdir(parents=True, exist_ok=True)
    for path in output.glob('*.png'):
        shutil.copy2(path, document_figures / path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--document-figures', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output, args.document_figures)
