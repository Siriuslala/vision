"""Deterministic candidate windows for teacher selection without coordinate guessing."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def candidate_windows(image: Image.Image, source_boxes: list | None = None, table: bool = False, chart: bool = False) -> tuple[list[list[int]], str]:
    width, height = image.size
    if chart:
        # Keep category labels and scales with chart marks. Small square tiles
        # frequently sever this association and cannot support chart answers.
        return [[0,0,(width*3)//5,height],[(width*2)//5,0,width,height],
                [0,0,width,(height*3)//5],[0,(height*2)//5,width,height]], 'chart_context_halves'
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    # Thin full-width rules delimit many rendered table rows. Broad flat backgrounds
    # are excluded; detection is deliberately restricted to near-uniform scanlines.
    scan = pixels[:, max(1, width//100):max(2, width-width//100)]
    mask = (scan.std(axis=1).max(axis=1) < 3) & (scan.mean(axis=(1, 2)) < 240)
    runs = []
    start = None
    for y, value in enumerate(list(mask)+[False]):
        if value and start is None:
            start = y
        elif not value and start is not None:
            if y-start <= 3:
                runs.append((start+y-1)//2)
            start = None
    if 4 <= len(runs) <= 24 and runs[-1]-runs[0] >= height*.6:
        edges = sorted(set([0]+runs+[height]))
        rows = [[0, max(0,a-1), width, min(height,b+2)] for a,b in zip(edges,edges[1:]) if b-a >= 8]
        if 3 <= len(rows) <= 16:
            return rows, 'detected_table_rows'
    if table:
        # Borderless tables often have separate horizontal bands of dark text.
        ink = (pixels.mean(axis=2)<100).mean(axis=1) > .01
        bands = []
        start = None
        for y, value in enumerate(list(ink)+[False]):
            if value and start is None:
                start = y
            elif not value and start is not None:
                if y-start >= 3:
                    bands.append((start,y))
                start = None
        if 2 <= len(bands) <= 16 and max(b-a for a,b in bands) <= height*.3:
            edges = [0]+[(a[1]+b[0])//2 for a,b in zip(bands,bands[1:])]+[height]
            return [[0,a,width,b] for a,b in zip(edges,edges[1:])], 'detected_table_text_bands'
    boxes = []
    for box in source_boxes or []:
        values = [int(round(v)) for v in box]
        if 0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height:
            boxes.append(values)
    for y in (0, .25, .5):
        for x in (0, .25, .5):
            boxes.append([int(x*width), int(y*height), int((x+.5)*width), int((y+.5)*height)])
    boxes += [[0,0,width,(height*3)//5], [0,(height*2)//5,width,height],
              [0,0,(width*3)//5,height], [(width*2)//5,0,width,height]]
    unique = []
    for box in boxes:
        if box not in unique and box[0] < box[2] and box[1] < box[3]:
            unique.append(box)
    return unique, 'source_boxes_and_overlapping_grid' if source_boxes else 'overlapping_grid'


def catalog_prompt(query: str, boxes: list[list[int]], method: str) -> str:
    return ('Solve the question from the supplied images. Image 1 is the full original. '
        'Image 2 is a contact sheet of candidate observations, each explicitly labelled WINDOW 1, WINDOW 2, etc. '
        'A window_id refers to the visible WINDOW label, NOT the image number. '
        'Choose only observations that provide necessary evidence. For comparing distinct objects or table rows, '
        'observe the relevant regions separately when useful; do not repeat equivalent evidence. '
        'Each selected window must actually show the complete relevant object or readable text. '
        'Write a correct, natural reasoning sequence: after each observation explain the finding, '
        'then combine findings and answer. No intentional mistakes or self-correction. '
        'In findings and conclusions describe visual contents; do not refer to temporary WINDOW numbers from the proposal sheet. '
        'Use at most six observations; use steps=[] if no extra observation is useful. '
        'Return ONLY one JSON object, with no prose outside it: '
        '{"answerable":true,"answer":"short answer; option letter when requested",'
        '"steps":[{"window_id":1,"finding":"specific observed evidence and its role"}],'
        '"conclusion":"combine the observations to justify the answer"}. '
        'Do not invent window IDs or coordinates. Original-image pixel boxes for these windows: '
        +str({i+1: box for i,box in enumerate(boxes)})+'. Proposal method: '+method+'. Question: '+query)


def contact_sheet(image: Image.Image, boxes: list[list[int]], large: bool = False) -> Image.Image:
    columns = 2 if large else 1 if max(box[3]-box[1] for box in boxes) < image.height*.4 else 3
    columns = min(columns,len(boxes))
    cell_width = 640 if large else 440 if columns==1 else 320
    previews = []
    for box in boxes:
        crop = image.crop(box).convert('RGB')
        scale = min(cell_width/crop.width, (520 if large else 260)/crop.height)
        previews.append(crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale)))))
    cell_height = max(p.height for p in previews)+32
    sheet = Image.new('RGB',(columns*cell_width,((len(previews)+columns-1)//columns)*cell_height),'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for index,preview in enumerate(previews):
        x,y = (index%columns)*cell_width,(index//columns)*cell_height
        draw.text((x+4,y+3),f'WINDOW {index+1}',fill='black',font=font)
        sheet.paste(preview,(x,y+30))
    return sheet
