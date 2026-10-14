"""BabyVision correspondence check for experiment 1.3 (design_decomposition.md, section 7).

Commands:

  annotate -- derive the instance grid of the 16 "Find the different" items from the manual
              grid specification below and write the annotation JSON plus overlay images
  build    -- construct T2 / T3 / T5 / T6 analogue cases into <output>/babyvision_cases.jsonl

Annotation method.  For every item the annotator (by visual inspection) fixed the number of
rows and columns and, where frames, titles or borders surround the grid, the pixel extent of
the instance grid.  The extent of the remaining items is the bounding box of all foreground
pixels (pixels whose max channel difference from the background colour exceeds 60).  The
extent is split into rows x cols equal cells; because the items are equally spaced, every
cell boundary falls in the blank gap between neighbouring instances.  The tight box of each
instance is the foreground bounding box inside its cell.  Overlays of all boxes are saved for
visual verification, and the row/column of the answer comes from the benchmark answer.

Images are padded (never resized) with the background colour on the right and bottom to a
multiple of 32 pixels, so the processor keeps the original pixel geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from analysis_tools.behavior.decomposition import (
    ANSWER_YES_NO, JSON_ONLY, LOCATE_SUFFIX, ORDINAL_NOTE, read_jsonl, sha256_file, write_jsonl,
)

ANNOTATION_VERSION = 'babyvision-find-different-grid-v1'
FOREGROUND_THRESHOLD = 60
# taskId -> (rows, cols, manual grid extent [x0, y0, x1) half-open or None, item noun, note)
GRID_SPEC: dict[int, tuple] = {
    445: (7, 7, None, 'tiger', ''),
    464: (25, 10, None, 'word', ''),
    507: (11, 28, None, 'letter', ''),
    525: (19, 10, None, 'word', 'Ignore color.'),
    534: (13, 20, (20, 14, 578, 477), 'lock', ''),
    638: (18, 22, None, 'letter', ''),
    779: (16, 10, (17, 8, 596, 548), 'group of three hands', ''),
    879: (14, 9, None, 'word', ''),
    910: (16, 8, None, 'group of three faces', ''),
    984: (10, 16, (40, 25, 700, 565), 'letter', ''),
    986: (15, 7, None, 'group of three faces', ''),
    988: (14, 10, None, 'number', ''),
    4152: (8, 13, (25, 15, 532, 454), 'tree', ''),
    6161: (16, 8, (47, 69, 564, 560), 'group of three faces', ''),
    6164: (17, 9, (47, 70, 581, 561), 'group of three fruits', ''),
    6165: (10, 10, None, 'animal', ''),
}
# Item 984: the extent marks the white panel; its foreground bounding box is used inside it.
PANEL_ONLY = {984}


def background(a: np.ndarray) -> np.ndarray:
    colors, counts = np.unique(a.reshape(-1, 3) // 8, axis=0, return_counts=True)
    return colors[counts.argmax()] * 8 + 4


def parse_answer(text: str) -> tuple[int, int]:
    nums = [int(v) for v in re.findall(r'\d+', text)]
    if len(nums) != 2:
        raise ValueError(f'Cannot parse row/column answer {text!r}')
    return nums[0], nums[1]


def annotate_item(image: Image.Image, rows: int, cols: int, extent, panel_only: bool) -> dict:
    a = np.asarray(image.convert('RGB')).astype(int)
    if extent is not None:
        x0, y0, x1, y1 = extent
        sub = a[y0:y1, x0:x1]
    else:
        x0, y0, sub = 0, 0, a
    bg = background(sub)
    fg = np.abs(sub - bg).max(-1) > FOREGROUND_THRESHOLD
    if extent is None or panel_only:
        ys, xs = np.nonzero(fg)
        gx0, gy0, gx1, gy1 = x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1
    else:
        gx0, gy0, gx1, gy1 = extent
    full = np.abs(a - bg).max(-1) > FOREGROUND_THRESHOLD
    cells, tight = [], []
    for r in range(rows):
        for c in range(cols):
            cx0 = int(round(gx0 + (gx1 - gx0) * c / cols))
            cx1 = int(round(gx0 + (gx1 - gx0) * (c + 1) / cols))
            cy0 = int(round(gy0 + (gy1 - gy0) * r / rows))
            cy1 = int(round(gy0 + (gy1 - gy0) * (r + 1) / rows))
            yy, xx = np.nonzero(full[cy0:cy1, cx0:cx1])
            if not len(xx):
                raise ValueError(f'Empty cell ({r}, {c})')
            cells.append([cx0, cy0, cx1, cy1])
            tight.append([cx0 + int(xx.min()), cy0 + int(yy.min()), cx0 + int(xx.max()) + 1, cy0 + int(yy.max()) + 1])
    return {'grid_extent': [int(gx0), int(gy0), int(gx1), int(gy1)], 'background': [int(v) for v in bg],
            'cell_boxes': cells, 'tight_boxes': tight}


def load_rows(babyvision: Path) -> dict[int, dict]:
    rows = [json.loads(l) for l in (babyvision / 'train.jsonl').read_text().splitlines() if l.strip()]
    chosen = {r['taskId']: r for r in rows if r['subtype'] == 'Find the different'}
    if set(chosen) != set(GRID_SPEC):
        raise ValueError(f'Find-the-different items differ from the annotated set: {sorted(chosen)}')
    return chosen


def annotate(babyvision: Path, annotation: Path, overlay_dir: Path) -> None:
    rows = load_rows(babyvision)
    items = []
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for tid, (nr, nc, extent, noun, note) in GRID_SPEC.items():
        row = rows[tid]
        path = babyvision / row['image']
        image = Image.open(path).convert('RGB')
        grid = annotate_item(image, nr, nc, extent, tid in PANEL_ONLY)
        ar, ac = parse_answer(row['blankAns'])
        items.append({'taskId': tid, 'image': row['image'], 'image_sha256': sha256_file(path), 'size': list(image.size),
                      'rows': nr, 'cols': nc, 'answer_rc': [ar - 1, ac - 1], 'item_noun': noun, 'note': note,
                      'manual_extent': list(extent) if extent else None, **grid})
        draw = ImageDraw.Draw(image)
        for k, box in enumerate(grid['tight_boxes']):
            color = (255, 0, 255) if k == (ar - 1) * nc + (ac - 1) else (0, 190, 0)
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=color)
        image.save(overlay_dir / f'{tid}.png')
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text(json.dumps({'version': ANNOTATION_VERSION, 'method': __doc__.split('Annotation method.')[1].split('Images are')[0].strip(),
                                      'foreground_threshold': FOREGROUND_THRESHOLD, 'items': items}, indent=1) + '\n')
    print(f'wrote {annotation} ({len(items)} items)')


def pad32(image: Image.Image, fill) -> Image.Image:
    w, h = image.size
    W, H = -(-max(w, 256) // 32) * 32, -(-max(h, 256) // 32) * 32
    canvas = Image.new('RGB', (W, H), tuple(fill))
    canvas.paste(image, (0, 0))
    return canvas


def save(image: Image.Image, output: Path, rel: str) -> dict:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    data = buf.getvalue()
    path = output / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Existing image {rel} differs.')
    path.write_bytes(data)
    return {'path': rel, 'sha256': hashlib.sha256(data).hexdigest(), 'image_id': rel.removesuffix('.png').replace('/', ':')}


def build(babyvision: Path, annotation: Path, output: Path, seed: int) -> None:
    ann = json.loads(annotation.read_text())
    cases = []
    for item in ann['items']:
        tid, nr, nc, noun = item['taskId'], item['rows'], item['cols'], item['item_noun']
        src = babyvision / item['image']
        if sha256_file(src) != item['image_sha256']:
            raise ValueError(f'Image hash changed for {tid}')
        image = Image.open(src).convert('RGB')
        bg = item['background']
        rng = np.random.default_rng(seed + tid)
        target = tuple(item['answer_rc'])
        t_index = target[0] * nc + target[1]
        others = [k for k in range(nr * nc) if k != t_index]
        normals = [divmod(int(k), nc) for k in rng.choice(others, 4, replace=False)]
        group = {'taskId': tid, 'grid_rows': nr, 'grid_cols': nc, 'kind': 'babyvision', 'level': 0, 'grid': nr * nc,
                 'template': noun, 'site': 0, 'rep': 0, 'record_id': f'bv{tid}', 'magnitude': None}
        base = save(pad32(image, bg), output, f'babyvision/{tid}.png')
        truth = {'rows': nr, 'cols': nc, 'target_rc': list(target)}
        note = (' ' + item['note']) if item['note'] else ''
        plural = noun + ('es' if noun.endswith('s') else 's') if not noun.startswith('group') else 'groups'

        def add(task, rec, prompt, variant, fmt, extra):
            cases.append({'id': f'babyvision|{task}|{tid}|{variant}|{fmt}', 'stage': 'babyvision', 'task': task,
                          'format': fmt, 'wording': 'primary', 'variant': variant, 'image': rec['path'],
                          'image_sha256': rec['sha256'], 'image_id': rec['image_id'], 'prompt': prompt,
                          'thinking': False, 'truth': {**truth, **extra}, **group})
        # T3: answer instance and three normal instances, by row/column text and by a blue box.
        for k, rc in enumerate([target] + normals[:3]):
            which = 'target' if k == 0 else f'normal{k}'
            yes = k != 0
            q = (f'Is the {noun} in row {rc[0] + 1}, column {rc[1] + 1} exactly identical to the other {plural}? '
                 + ORDINAL_NOTE + note)
            add('T3', base, q + ANSWER_YES_NO, f'rowcol_{which}', 'yes_no', {'answer_yes': yes, 'designated_rc': list(rc)})
            marked = image.copy()
            x0, y0, x1, y1 = item['cell_boxes'][rc[0] * nc + rc[1]]
            ImageDraw.Draw(marked).rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 0, 255), width=2)
            rec = save(pad32(marked, bg), output, f'babyvision/{tid}_box_{which}.png')
            q = f'Is the {noun} inside the blue box exactly identical to the other {plural}?' + note
            add('T3', rec, q + ANSWER_YES_NO, f'box_{which}', 'yes_no', {'answer_yes': yes, 'designated_rc': list(rc)})
        # T2: crops placed side by side; (answer, normal) and (normal, normal) pairs.
        def crop(rc):
            x0, y0, x1, y1 = item['cell_boxes'][rc[0] * nc + rc[1]]
            return image.crop((x0, y0, x1, y1))
        for variant, pair, yes in (('neg', (target, normals[0]), False), ('pos', (normals[1], normals[2]), True)):
            tiles = [crop(rc) for rc in pair]
            if rng.integers(0, 2):
                tiles = tiles[::-1]
            gap = 16
            w = sum(t.width for t in tiles) + 3 * gap
            h = max(t.height for t in tiles) + 2 * gap
            canvas = Image.new('RGB', (w, h), tuple(bg))
            x = gap
            for t in tiles:
                canvas.paste(t, (x, gap + (h - 2 * gap - t.height) // 2))
                x += t.width + gap
            rec = save(pad32(canvas, bg), output, f'babyvision/{tid}_pair_{variant}.png')
            q = f'Are the two {plural} exactly identical?' + note
            add('T2', rec, q + ANSWER_YES_NO, variant, 'yes_no', {'answer_yes': yes})
        # T5: the original task in the row/column JSON format; T6: the answer instance recoloured.
        q5 = f'One {noun} differs from all the others.' + note
        add('T5', base, q5 + LOCATE_SUFFIX['row_column'] + JSON_ONLY, 'target', 'row_column', {})
        a = np.asarray(image).astype(int).copy()
        x0, y0, x1, y1 = item['cell_boxes'][t_index]
        cell = a[y0:y1, x0:x1]
        mask = np.abs(cell - np.array(bg)).max(-1) > FOREGROUND_THRESHOLD
        red_bg = bg[0] > 200 and bg[1] < 60 and bg[2] < 60
        color, cname = ((0, 0, 255), 'blue') if red_bg else ((255, 0, 0), 'red')
        cell[mask] = color
        recol = Image.fromarray(a.astype(np.uint8))
        rec = save(pad32(recol, bg), output, f'babyvision/{tid}_{cname}.png')
        q6 = f'One {noun} is drawn in {cname}.'
        add('T6', rec, q6 + LOCATE_SUFFIX['row_column'] + JSON_ONLY, 'target', 'row_column', {'color': cname})
    write_jsonl(output / 'babyvision_cases.jsonl', cases)
    print(f'{len(cases)} BabyVision cases')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('annotate', 'build'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--babyvision', type=Path, required=True)
    p.add_argument('--annotation', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.command == 'annotate':
        annotate(args.babyvision, args.annotation, args.output / 'babyvision_overlays')
    else:
        build(args.babyvision, args.annotation, args.output, args.seed)


if __name__ == '__main__':
    main()
