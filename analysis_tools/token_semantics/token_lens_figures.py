"""Summarize fixed-case raw logit lens outputs without rerunning inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont
import os


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text()) for p in sorted(args.input.glob('*/tokens.json'))]
    if len(records) != 12:
        raise ValueError('Expected all 12 fixed image cases.')
    layers = list(range(24, 33))
    rows = []
    for layer in layers:
        sample_metrics = []
        for case in records:
            tokens = case['layers'][f'layer_{layer}']['tokens']
            sample_metrics.append([
                np.mean([t['top_probs'][0] for t in tokens]),
                np.mean([t['entropy'] for t in tokens]),
                np.mean([t['top_tokens'][0].startswith('<|') for t in tokens]),
            ])
        rows.append({'layer': layer, 'top1_probability': float(np.mean(sample_metrics, axis=0)[0]),
                     'entropy': float(np.mean(sample_metrics, axis=0)[1]),
                     'special_top1_fraction': float(np.mean(sample_metrics, axis=0)[2]),
                     'per_image': {r['id']: m for r, m in zip(records, sample_metrics)}})
    (args.output / 'layer_metrics.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False)+'\n')
    font_path = os.environ['TOKEN_MAP_FONT']
    label_font = FontProperties(fname=font_path, size=11)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, key, title, ymax in zip(axes,
            ['top1_probability', 'entropy', 'special_top1_fraction'],
            ['Top-1 概率', '词表 entropy（nats）', '特殊 token 的 top-1 比例'], [1., 7., .25]):
        ax.plot(layers, [r[key] for r in rows], marker='o', color='#245e91')
        ax.set_xticks(layers)
        ax.set_ylim(0, ymax)
        ax.set_xlabel('Decoder 层编号', fontproperties=label_font)
        ax.set_title(title, fontproperties=label_font)
        ax.grid(alpha=.2)
    fig.suptitle('Raw logit lens：12幅图像等权平均', fontproperties=label_font)
    fig.tight_layout()
    fig.savefig(args.output / 'layer_distribution.png', dpi=180)
    fig.savefig(args.output / 'layer_distribution.pdf')
    plt.close(fig)
    font = ImageFont.truetype(font_path, 22)
    parts = [Image.open(args.input/'people_normal'/f'layer_{layer}_top{rank}.png').convert('RGB')
             for layer in (31, 32) for rank in (1, 2, 3)]
    cw, ch = parts[0].size
    sheet = Image.new('RGB', (cw*3, ch*2+65), 'white')
    ImageDraw.Draw(sheet).text((12, 12), '正常小人图：上排 decoder 第31层，下排第32层；各列为 top-1 / top-2 / top-3。', font=font, fill='black')
    for i, part in enumerate(parts):
        sheet.paste(part, ((i%3)*cw, (i//3)*ch+65))
    sheet.save(args.output/'people_last_two_layers.png')
    sample_sheet = Image.new('RGB', (4*360, 3*310), 'white')
    draw = ImageDraw.Draw(sample_sheet)
    for i, case in enumerate(records):
        im = Image.open(args.input/case['id']/'image.png').convert('RGB')
        im.thumbnail((342, 262))
        x, y = (i%4)*360, (i//4)*310
        sample_sheet.paste(im, (x+(360-im.width)//2, y+35))
        draw.text((x+10,y+3), case['id'], font=font, fill='black')
    sample_sheet.save(args.output/'sample_images.png')


if __name__ == '__main__':
    main()
