"""Experiment 3.11 round 3: roles of the edge-reading attention heads (case building and GPU stages).

Results: docs/analysis/position/results_2026-11-03_edge_heads.md. Builds on 3.11 round 2
(coordinate_flow_natural.py), which found that on RefCOCO the full-attention heads that read the target at the
first-digit query of a coordinate value split into min-edge readers (left/top) and max-edge readers
(right/bottom), and that L20 H11 reads a semantically salient part.

Round 3 asks three questions:
  1. Which edge, edge pair or corner does each head read at each emitted value? Values: the x and y of a
     point answer and x_min, y_min, x_max, y_max of a box answer. The attention row of every full-attention
     head is recorded at the query that predicts the first digit of every value (the position before the digit).
  2. Does the picture hold across many object categories and sizes, and which part of a non-person object
     attracts L20 H11? Objects come from four local sources with instance masks; three of them also have
     part masks (PACO-LVIS and Pascal-Part).
  3. What does L20 H11 contribute on natural images? 'path' patches the H11 output (and controls) at the
     answer positions from a run in which the expression names another object, and records how the attention
     of the later full-attention heads (layers 24, 28, 32) moves between the two objects.

Groups (case['group']):
  rc      the 300 RefCOCO point questions of 2.5e / 3.11 round 2 (persons, animals, other), COCO masks.
  paco    PACO-LVIS val/test objects (COCO train2017 images; furniture, tableware, electronics, wearables,
          small objects, cars, bicycles, dogs) with object and part polygons. The category is unique in the
          image and exhaustively annotated, so 'the <category>' names one object.
  vocpart Pascal-Part (VOC2010) objects of the 14 part-annotated classes (person, six animals, five vehicles,
          bottle, potted plant), one instance of the class per image; part masks (RLE).
  stuff   COCO-Stuff semantic maps (train2017) for categories absent from the other sources (food, large
          animals, large vehicles, furniture, appliances, sport objects). A class region is accepted as one
          instance when one 4-connected component holds >= 97% of its pixels (heuristic, stated in the doc).

Stages (--stages; GPU stages run in one process and resume by record id):
  cases   CPU: object selection, 512-pixel images, mask and part-label PNGs, prompts, what pairs.
  gen     greedy answers to the point and the box prompt, then a teacher-forced forward on each answer that
          records the image attention of all 128 full-attention heads at every value's first-digit query.
  means   mean full-attention head outputs over answer positions (per group family) for mean ablation.
  ablate  greedy point answers with mean-ablated head sets at the answer positions.
  path    what-pair patching of L20 H11 and controls; later-layer attention and first-digit logit differences.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
import torch

from analysis_tools.common import qwen_runtime as qr
from analysis_tools.position import coordinate_flow as cf
from analysis_tools.position import coordinate_flow_natural as cfn
from analysis_tools.position import coordinate_readout as cr
from analysis_tools.position.position_probe_refinement import resize_to
from analysis_tools.representation import sees_right as sr

FULL_LAYERS = cr.FULL_LAYERS
DN_LAYERS = cfn.DN_LAYERS
TP = cr.TOKEN_PIXELS
read_jsonl, append_jsonl, done_ids = sr.read_jsonl, sr.append_jsonl, sr.done_ids
FORMATS = ('point', 'bbox')
MIN_AREA, MAX_AREA = 0.005, 0.6          # object area / image area

# COCO category ids (COCO-Stuff label = id - 1) of the categories taken from COCO-Stuff, by family.
STUFF_CLASSES = {
    'food': {52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
             58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake'},
    'animal': {22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe'},
    'vehicle': {5: 'airplane', 7: 'train', 8: 'truck', 9: 'boat'},
    'furniture': {63: 'couch', 65: 'bed', 70: 'toilet', 82: 'refrigerator', 79: 'oven', 81: 'sink'},
    'small_object': {28: 'umbrella', 38: 'kite', 41: 'skateboard', 42: 'surfboard', 88: 'teddy bear',
                     11: 'fire hydrant', 13: 'stop sign', 33: 'suitcase', 43: 'tennis racket'},
}
VOC_FAMILY = {'person': 'person', 'bird': 'animal', 'cat': 'animal', 'cow': 'animal', 'dog': 'animal',
              'horse': 'animal', 'sheep': 'animal', 'aeroplane': 'vehicle', 'bicycle': 'vehicle', 'bus': 'vehicle',
              'car': 'vehicle', 'motorbike': 'vehicle', 'bottle': 'small_object', 'pottedplant': 'furniture'}
VOC_NAME = {'aeroplane': 'airplane', 'motorbike': 'motorcycle', 'pottedplant': 'potted plant'}
PACO_FAMILY = {
    'furniture': ['chair', 'table', 'bench', 'stool', 'lamp', 'pillow', 'mirror', 'clock', 'ladder', 'fan', 'crate',
                  'trash_can', 'television_set', 'microwave_oven'],
    'tableware': ['bowl', 'cup', 'mug', 'plate', 'bottle', 'glass_(drink_container)', 'jar', 'knife', 'spoon',
                  'pan_(for_cooking)', 'kettle', 'tray', 'can', 'blender', 'vase', 'napkin'],
    'electronics': ['laptop_computer', 'cellular_telephone', 'remote_control', 'mouse_(computer_equipment)',
                    'telephone', 'earphone', 'calculator'],
    'wearable': ['hat', 'shoe', 'helmet', 'handbag', 'belt', 'watch', 'scarf', 'sweater', 'slipper_(footwear)',
                 'wallet'],
    'vehicle': ['car_(automobile)', 'bicycle'],
    'animal': ['dog'],
}
PACO_NAME = {'car_(automobile)': 'car', 'glass_(drink_container)': 'drinking glass', 'pan_(for_cooking)': 'frying pan',
             'mouse_(computer_equipment)': 'computer mouse', 'cellular_telephone': 'cell phone',
             'laptop_computer': 'laptop', 'television_set': 'television', 'microwave_oven': 'microwave oven',
             'slipper_(footwear)': 'slipper', 'trash_can': 'trash can', 'remote_control': 'remote control',
             'tape_(sticky_cloth_or_paper)': 'tape', 'drum_(musical_instrument)': 'drum'}


def prompt_for(expr: str, fmt: str) -> str:
    """The 2.5e / 3.11 RefCOCO prompt with the point or box format suffix."""
    return sr.REFCOCO_QUERY.format(expr=expr) + cf.FORMAT_SUFFIX[fmt]


# ----------------------------------------------------------------------------- masks

def rle_decode(counts: str | list[int], h: int, w: int) -> np.ndarray:
    """COCO RLE (compressed string or uncompressed list, column-major) to a (h, w) bool mask."""
    if isinstance(counts, str):
        s, p, cnts = counts.encode(), 0, []
        while p < len(s):
            x, k, more = 0, 0, True
            while more:
                c = s[p] - 48
                x |= (c & 0x1f) << (5 * k)
                more = bool(c & 0x20)
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(cnts) > 2:
                x += cnts[-2]
            cnts.append(x)
    else:
        cnts = list(counts)
    flat = np.zeros(h * w, bool)
    pos, val = 0, False
    for n in cnts:
        if val:
            flat[pos:pos + n] = True
        pos += n
        val = not val
    if pos != h * w:
        raise ValueError(f'RLE length {pos} != {h * w}')
    return flat.reshape(w, h).T


def polygons_mask(polys: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    """Union of COCO polygons ([x0, y0, x1, y1, ...]) rasterized at size = (W, H)."""
    im = Image.new('L', size, 0)
    d = ImageDraw.Draw(im)
    for poly in polys:
        if len(poly) >= 6:
            d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
    return np.asarray(im, bool)


def components(mask: np.ndarray) -> np.ndarray:
    """4-connected component labels (0 = background, 1.. = components) by min-label propagation."""
    idx = np.arange(1, mask.size + 1).reshape(mask.shape)
    lab = np.where(mask, idx, 0)
    big = mask.size + 1
    while True:
        cur = np.where(mask, lab, big)
        nb = cur.copy()
        nb[1:] = np.minimum(nb[1:], cur[:-1])
        nb[:-1] = np.minimum(nb[:-1], cur[1:])
        nb[:, 1:] = np.minimum(nb[:, 1:], cur[:, :-1])
        nb[:, :-1] = np.minimum(nb[:, :-1], cur[:, 1:])
        new = np.where(mask, nb, 0)
        if np.array_equal(new, lab):
            break
        lab = new
    _, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(mask.shape)


def mask_box(mask: np.ndarray) -> list[float]:
    """Tight half-open pixel box [x0, y0, x1, y1] of a bool mask."""
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.BILINEAR)) >= 128


def scale_box(b, src_size, in_size) -> list[float]:
    sx, sy = in_size[0] / src_size[0], in_size[1] / src_size[1]
    return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]


# ----------------------------------------------------------------------------- case sources

def paco_candidates(paco_dir: Path) -> list[dict]:
    """Objects whose category is unique and exhaustively annotated in its image; parts attached by obj_ann_id."""
    allowed = {n: fam for fam, names in PACO_FAMILY.items() for n in names}
    out = []
    for split in ('val', 'test'):
        d = json.loads((paco_dir / f'paco_lvis_v1_{split}.json').read_text())
        cats = {c['id']: c['name'] for c in d['categories']}
        imgs = {im['id']: im for im in d['images']}
        objs, parts = {}, {}
        for a in d['annotations']:
            name = cats[a['category_id']]
            if ':' in name:
                parts.setdefault(a['obj_ann_id'], []).append({'name': name.split(':')[1], 'seg': a['segmentation']})
            else:
                objs.setdefault(a['image_id'], []).append({**a, 'name': name})
        for img_id, lst in objs.items():
            im = imgs[img_id]
            counts: dict[str, int] = {}
            for a in lst:
                counts[a['name']] = counts.get(a['name'], 0) + 1
            nonex = {cats[c] for c in im.get('not_exhaustive_category_ids', [])}
            uniq = [a for a in lst if counts[a['name']] == 1 and a['name'] not in nonex]
            for a in uniq:
                frac = a['area'] / (im['width'] * im['height'])
                if a['name'] not in allowed or not MIN_AREA <= frac <= MAX_AREA or a['id'] not in parts:
                    continue
                others = [o for o in uniq if o['id'] != a['id'] and o['area'] / (im['width'] * im['height']) >= MIN_AREA]
                out.append({'source': 'paco', 'split': split, 'image_file': im['file_name'], 'image_id': img_id,
                            'size': [im['width'], im['height']], 'category': a['name'],
                            'name': PACO_NAME.get(a['name'], a['name'].replace('_', ' ')), 'family': allowed[a['name']],
                            'seg': a['segmentation'], 'bbox_xywh': a['bbox'], 'area_frac': frac,
                            'parts': parts[a['id']],
                            'others': [{'name': PACO_NAME.get(o['name'], o['name'].replace('_', ' ')),
                                        'bbox_xywh': o['bbox']} for o in others]})
    return out


def vocpart_candidates(voc_dir: Path, part_json: Path) -> list[dict]:
    """Pascal-Part objects: the only instance of a part-annotated class in its image."""
    d = json.loads(part_json.read_text())
    cats = {c['id']: c['name'] for c in d['categories']}
    parts: dict[int, list] = {}
    for a in d['annotations']:
        parts.setdefault(a['image_id'], []).append(a)
    out = []
    for im in d['images']:
        stem = im['file_name'].rsplit('.', 1)[0]
        root = ET.parse(voc_dir / 'Annotations' / f'{stem}.xml').getroot()
        objs = root.findall('object')
        names = [o.find('name').text for o in objs]
        for k, (o, cls) in enumerate(zip(objs, names)):
            if cls not in VOC_FAMILY or names.count(cls) != 1:
                continue
            ps = [p for p in parts.get(im['id'], []) if cats[p['category_id']].split(':')[0] == cls]
            if not ps:
                continue
            bb = o.find('bndbox')
            box = [float(bb.find('xmin').text) - 1, float(bb.find('ymin').text) - 1,
                   float(bb.find('xmax').text), float(bb.find('ymax').text)]
            frac = (box[2] - box[0]) * (box[3] - box[1]) / (im['width'] * im['height'])
            if not MIN_AREA <= frac <= MAX_AREA * 1.2:      # box area; masks are smaller than boxes
                continue
            out.append({'source': 'vocpart', 'image_file': im['file_name'], 'size': [im['width'], im['height']],
                        'category': cls, 'name': VOC_NAME.get(cls, cls), 'family': VOC_FAMILY[cls], 'box': box,
                        'object_index': k + 1, 'area_frac': frac,
                        'parts': [{'name': cats[p['category_id']].split(':')[1], 'seg': p['segmentation']} for p in ps]})
    return out


def stuff_instance(label_map: np.ndarray, label: int) -> np.ndarray | None:
    """The class region as one instance if one 4-connected component holds >= 97% of its pixels (stride-4 map)."""
    m = label_map == label
    frac = m.mean()
    if not MIN_AREA <= frac <= MAX_AREA:
        return None
    small = m[::4, ::4]
    lab = components(small)
    sizes = np.bincount(lab.ravel())[1:]
    if sizes.size == 0 or sizes.max() < 0.97 * sizes.sum():
        return None
    return m


# ----------------------------------------------------------------------------- case building

def save_case_images(img_dir: Path, cid: str, image: Image.Image, mask: np.ndarray | None,
                     parts: list[tuple[str, np.ndarray]]) -> dict:
    """512-pixel image (longest side, multiples of 32), mask PNG and part-label PNG; returns paths and sizes."""
    r = resize_to(image, 512)
    p = img_dir / f'{cid}.png'
    r.save(p)
    out = {'image_path': str(p), 'image_sha256': sr.sha256(p), 'source_size': list(image.size), 'canvas': list(r.size)}
    if mask is not None:
        m = resize_mask(mask, r.size)
        if not m.any():
            raise ValueError(f'Empty resized mask for {cid}')
        Image.fromarray(m.astype(np.uint8) * 255).save(img_dir / f'{cid}_mask.png')
        out['mask_path'] = str(img_dir / f'{cid}_mask.png')
    if parts:
        lab = np.zeros((r.size[1], r.size[0]), np.uint8)
        names = []
        for name, pm in parts:
            if name not in names:
                names.append(name)
            lab[resize_mask(pm, r.size)] = names.index(name) + 1
        Image.fromarray(lab).save(img_dir / f'{cid}_parts.png')
        out['parts_path'] = str(img_dir / f'{cid}_parts.png')
        out['part_names'] = names
    return out


def finish_case(cid: str, group: str, info: dict, name: str, family: str, category: str, box_src: list[float],
                extra: dict) -> dict:
    box = scale_box(box_src, info['source_size'], info['canvas'])
    expr = name if group == 'rc' else f'the {name}'
    return {'id': cid, 'group': group, 'set': group, 'family': family, 'category': category, 'expression': expr,
            'prompt': prompt_for(expr, 'point'), 'prompt_bbox': prompt_for(expr, 'bbox'),
            'truth': {'canvas': info['canvas'], 'box': box, 'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]},
            **{k: v for k, v in info.items() if k not in ('canvas',)}, **extra}


def stratified(cands: list[dict], key: str, per_key: int, rng: np.random.Generator) -> list[dict]:
    """At most per_key candidates per category (one per image), spread over area quartiles by round-robin."""
    by: dict[str, list] = {}
    for c in cands:
        by.setdefault(c[key], []).append(c)
    out = []
    for k in sorted(by):
        lst = [by[k][i] for i in rng.permutation(len(by[k]))]
        lst.sort(key=lambda c: c['area_frac'])
        bins = np.array_split(np.arange(len(lst)), min(4, len(lst)))
        picked, seen, r = [], set(), 0
        while len(picked) < per_key and any(len(b) > r for b in bins):
            for b in bins:
                if r < len(b) and len(picked) < per_key:
                    c = lst[b[r]]
                    if c['image_file'] not in seen:
                        picked.append(c)
                        seen.add(c['image_file'])
            r += 1
        out += picked
    return out


def stage_cases(out: Path, args) -> None:
    rng = np.random.default_rng(args.seed)
    img_dir = out / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    used_images: set[str] = set()

    # rc: the RefCOCO cases of 3.11 round 2 (images, boxes, masks and what pairs reused).
    from analysis_tools.position.coordinate_flow_natural_analysis import category as rc_category
    for c in json.loads((args.round2 / 'cases.json').read_text()):
        if c['group'] != 'rc':
            continue
        W, H = c['truth']['canvas']
        m = Image.new('L', (int(W), int(H)), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in c['mask']], fill=255)
        mp = img_dir / f"rc_{c['id']}_mask.png"
        m.save(mp)
        fam = rc_category(c['expression'])
        case = {'id': f"rc_{c['id']}", 'group': 'rc', 'set': 'rc', 'family': fam, 'category': f'refcoco_{fam}',
                'expression': c['expression'], 'prompt': c['prompt'], 'prompt_bbox': prompt_for(c['expression'], 'bbox'),
                'truth': c['truth'], 'image_path': c['image_path'], 'image_sha256': c['image_sha256'],
                'canvas': c['truth']['canvas'], 'mask_path': str(mp), 'others': c['others']}
        if case['prompt'] != prompt_for(c['expression'], 'point'):
            raise ValueError(f"Unexpected RefCOCO prompt for {c['id']}")
        if 'prompt_b' in c:
            case.update(expression_b=c['expression_b'], prompt_b=c['prompt_b'], truth_b=c['truth_b'])
        case['area_frac'] = float(np.asarray(m).astype(bool).mean())
        cases.append(case)

    # paco
    coco_dir = Path(args.coco_dir)
    # only train2017 COCO images are available locally; PACO objects on val2017 images are skipped
    pc = [o for o in paco_candidates(Path(args.paco_dir)) if (coco_dir / o['image_file']).exists()]
    pc = stratified(pc, 'category', args.per_category_paco, rng)
    for i, o in enumerate(pc):
        cid = f'paco_{i:04d}'
        image = Image.open(coco_dir / o['image_file']).convert('RGB')
        if list(image.size) != o['size']:
            raise ValueError(f'PACO size mismatch for {o["image_file"]}')
        mask = polygons_mask(o['seg'], image.size)
        parts = [(p['name'], polygons_mask(p['seg'], image.size) if isinstance(p['seg'], list)
                  else rle_decode(p['seg']['counts'], image.size[1], image.size[0])) for p in o['parts']]
        info = save_case_images(img_dir, cid, image, mask, parts)
        x, y, w, h = o['bbox_xywh']
        case = finish_case(cid, 'paco', info, o['name'], o['family'], o['category'], [x, y, x + w, y + h],
                           {'area_frac': o['area_frac'], 'source': {'file': o['image_file'], 'split': o['split']}})
        others = [{'box': scale_box([b['bbox_xywh'][0], b['bbox_xywh'][1], b['bbox_xywh'][0] + b['bbox_xywh'][2],
                                     b['bbox_xywh'][1] + b['bbox_xywh'][3]], info['source_size'], info['canvas']),
                   'name': b['name']} for b in o['others']]
        case['others'] = [b['box'] for b in others]
        dis = cfn.pick_distractor(case['truth']['box'], others, info['canvas'])
        if dis is not None:
            case.update(expression_b=f"the {dis['name']}", prompt_b=prompt_for(f"the {dis['name']}", 'point'),
                        truth_b={'box': dis['box'], 'center': [(dis['box'][0] + dis['box'][2]) / 2,
                                                              (dis['box'][1] + dis['box'][3]) / 2]})
        cases.append(case)
        used_images.add(o['image_file'].split('/')[-1].split('.')[0])

    # vocpart
    voc_dir = Path(args.voc_dir)
    vc = stratified(vocpart_candidates(voc_dir, Path(args.vocpart_json)), 'category', args.per_category_voc, rng)
    for i, o in enumerate(vc):
        cid = f'voc_{i:04d}'
        image = Image.open(voc_dir / 'JPEGImages' / o['image_file']).convert('RGB')
        W, H = image.size
        parts = [(p['name'], rle_decode(p['seg']['counts'], H, W)) for p in o['parts']]
        seg_path = voc_dir / 'SegmentationObject' / f"{o['image_file'].rsplit('.', 1)[0]}.png"
        mask, mask_src = None, 'none'
        if seg_path.exists():
            mask = np.asarray(Image.open(seg_path)) == o['object_index']
            mask_src = 'voc_instance'
        elif o['family'] in ('person', 'animal'):
            mask = np.any([pm for _, pm in parts], 0)       # body parts of persons and animals tile the object
            mask_src = 'part_union'
        if mask is not None and not mask.any():
            mask, mask_src = None, 'none'
        info = save_case_images(img_dir, cid, image, mask, parts)
        cases.append(finish_case(cid, 'vocpart', info, o['name'], o['family'], o['category'], o['box'],
                                 {'area_frac': o['area_frac'], 'mask_source': mask_src,
                                  'source': {'file': o['image_file']}}))

    # stuff: scan train2017 label maps in random order until the per-class quota is filled.
    label_of = {cid - 1: (name, fam) for fam, d in STUFF_CLASSES.items() for cid, name in d.items()}
    quota = {lab: args.per_category_stuff for lab in label_of}
    files = sorted(Path(args.stuff_dir).glob('*.png'))
    order = rng.permutation(len(files))
    n_stuff, scanned = 0, 0
    for j in order:
        if not any(quota.values()) or scanned >= args.stuff_scan_limit:
            break
        f = files[j]
        scanned += 1
        if f.stem in used_images:
            continue
        lm = np.asarray(Image.open(f))
        present = [l for l in np.unique(lm) if quota.get(int(l), 0) > 0]
        for lab in present:
            inst = stuff_instance(lm, int(lab))
            if inst is None:
                continue
            name, fam = label_of[int(lab)]
            image = Image.open(coco_dir / 'train2017' / f'{f.stem}.jpg').convert('RGB')
            if image.size != (lm.shape[1], lm.shape[0]):
                raise ValueError(f'COCO-Stuff size mismatch for {f.stem}')
            cid = f'stuff_{n_stuff:04d}'
            info = save_case_images(img_dir, cid, image, inst, [])
            cases.append(finish_case(cid, 'stuff', info, name, fam, name, mask_box(inst),
                                     {'area_frac': float(inst.mean()), 'source': {'file': f'train2017/{f.stem}.jpg'}}))
            quota[int(lab)] -= 1
            n_stuff += 1
            break                                            # one object per image
    for c in cases:
        c.setdefault('canvas', c['truth']['canvas'])
    (out / 'cases.json').write_text(json.dumps(cases, indent=1))
    summary: dict[str, dict] = {}
    for c in cases:
        s = summary.setdefault(c['group'], {'cases': 0, 'with_what_pair': 0, 'families': {}})
        s['cases'] += 1
        s['with_what_pair'] += int('prompt_b' in c)
        s['families'][c['family']] = s['families'].get(c['family'], 0) + 1
    summary['stuff_scanned_label_maps'] = scanned
    (out / 'cases_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def load_cases(out: Path) -> dict:
    return {c['id']: c for c in json.loads((out / 'cases.json').read_text())}


# ----------------------------------------------------------------------------- scoring

def score(case: dict, values: list[dict] | None, fmt: str) -> dict:
    """Point: inside the target box. Box: IoU with the target box >= 0.5 (both in input pixels)."""
    keys = ('x', 'y') if fmt == 'point' else ('x_min', 'y_min', 'x_max', 'y_max')
    if values is None or [v['key'] for v in values] != list(keys):
        return {'valid': False, 'correct': False}
    v = {d['key']: d['value'] for d in values}
    W, H = case['truth']['canvas']
    b = case['truth']['box']
    if fmt == 'point':
        px, py = v['x'] * W / 1000, v['y'] * H / 1000
        return {'valid': True, 'correct': bool(b[0] <= px <= b[2] and b[1] <= py <= b[3])}
    pb = [v['x_min'] * W / 1000, v['y_min'] * H / 1000, v['x_max'] * W / 1000, v['y_max'] * H / 1000]
    iou = cr.box_iou(pb, b) if pb[2] > pb[0] and pb[3] > pb[1] else 0.0
    return {'valid': True, 'correct': bool(iou >= 0.5), 'iou': round(float(iou), 4)}


# ----------------------------------------------------------------------------- attention capture

def forward_with_attention(rt, x: torch.Tensor, pos: torch.Tensor, queries: list[int], layers, logits_from: int
                           ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, float]]:
    """One batched forward; returns tail logits (B, n_tail, V), attention rows {layer: (B, nq, 16, L)} at the
    queries (shared by all rows) and the relative reconstruction error of each layer's attention output.

    The rows are recomputed from the captured attention inputs exactly as in coordinate_flow.attention_rows
    (q, k with q/k norm and rotary embedding, causal softmax), batched over rows. Any output-projection
    pre-hook that edits head outputs (patching) acts after the capture hook registered here (prepend=True), so
    the check compares the recomputed head outputs with the unedited o_proj input of every row.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    store, handles = {}, []
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn

        def pre(module, args, kwargs, layer=layer):
            store.setdefault(layer, {})['h'] = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            store[layer]['pe'] = kwargs['position_embeddings']

        def opre(module, args, layer=layer):
            store.setdefault(layer, {})['o_in'] = args[0]
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(mod.o_proj.register_forward_pre_hook(opre, prepend=True))
    try:
        logits = rt.model(inputs_embeds=x, position_ids=pos, use_cache=False,
                          logits_to_keep=slice(logits_from, None)).logits.float()
    finally:
        for h in handles:
            h.remove()
    attn, check = {}, {}
    for layer in layers:
        mod = rt.layers[layer - 1].self_attn
        h, (cos, sin), o_in = store[layer]['h'], store[layer]['pe'], store[layer]['o_in']
        dev = h.device
        B, L, _ = h.shape
        q_idx = torch.tensor(queries, device=dev)
        cos, sin = cos.to(dev), sin.to(dev)
        hd = mod.head_dim
        q, gate = torch.chunk(mod.q_proj(h[:, q_idx]).view(B, len(queries), -1, hd * 2), 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(h).view(B, L, -1, hd)).transpose(1, 2)
        v = mod.v_proj(h).view(B, L, -1, hd).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, q_idx], sin[:, q_idx])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k, v = repeat_kv(k, mod.num_key_value_groups), repeat_kv(v, mod.num_key_value_groups)
        sc = (q.float() @ k.float().transpose(2, 3)) * mod.scaling
        sc = sc.masked_fill(torch.arange(L, device=dev)[None, :] > q_idx[:, None], float('-inf'))
        a = sc.softmax(-1)                                                   # (B, 16, nq, L)
        o = ((a @ v.float()) * torch.sigmoid(gate.float()).transpose(1, 2)).transpose(1, 2).reshape(B, len(queries), -1)
        ref = o_in[:, q_idx.to(o_in.device)].to(dev).float()
        err = ((o - ref).abs().amax((1, 2)) / ref.abs().amax((1, 2))).max()
        check[layer] = float(err)
        attn[layer] = a.transpose(1, 2).cpu()                                # (B, nq, 16, L)
    return logits, attn, check


# ----------------------------------------------------------------------------- gen + attention

@torch.inference_mode()
def stage_gen(rt, out: Path, args) -> None:
    """Greedy point and box answers, then per answer the first-digit attention of all full-attention heads."""
    path = out / 'gen.jsonl'
    (out / 'attn').mkdir(exist_ok=True)
    finished = done_ids(path)
    cases = load_cases(out)
    items = [(c, fmt) for c in cases.values() if c['group'] in args.groups for fmt in FORMATS
             if f"{c['id']}:{fmt}" not in finished]
    for i in range(0, len(items), args.prep_chunk):
        chunk = items[i:i + args.prep_chunk]
        preps = [cr.prepare(rt, sr.load_image(c), c['prompt'] if fmt == 'point' else c['prompt_bbox'])
                 for c, fmt in chunk]
        by_len: dict[int, list] = {}
        for it, p in zip(chunk, preps):
            by_len.setdefault(len(p.ids), []).append((it, p))
        for L, group in by_len.items():
            for j in range(0, len(group), args.batch_size):
                part = group[j:j + args.batch_size]
                res = cr.run_conditions(rt, [cr.embeds(rt, p) for _, p in part], [p.pos for _, p in part],
                                        args.max_new_tokens, batch_size=len(part))
                rows = []
                for ((c, fmt), prep), r in zip(part, res):
                    vals = cf.answer_values(rt, r['ids'], fmt)
                    sc = score(c, vals, fmt)
                    rec = {'id': f"{c['id']}:{fmt}", 'case': c['id'], 'format': fmt, 'group': c['group'],
                           'text': r['text'], 'ids': r['ids'], 'prompt_len': L, 'batch_size': len(part),
                           'grid': list(prep.grid), 'values': vals, 'score': sc}
                    if sc['valid']:
                        x, pos = cr.teacher_forced(rt, prep, r['ids'][:-1])
                        q = cf.first_digit_queries(vals, L)
                        _, att, chk = forward_with_attention(rt, x, pos, q, FULL_LAYERS, x.shape[1] - 1)
                        s, e = prep.span['start'], prep.span['end']
                        a = np.stack([att[l][0, :, :, s:e].numpy() for l in FULL_LAYERS], 1)   # (nq, 8, 16, Nv)
                        np.savez_compressed(out / 'attn' / f"{c['id']}_{fmt}.npz", attn=a.astype(np.float16))
                        rec['attn_check'] = max(chk.values())
                    rows.append(rec)
                append_jsonl(path, rows)
        print(f'gen {i + len(chunk)}/{len(items)}', flush=True)
        del preps
        torch.cuda.empty_cache()


@torch.inference_mode()
def stage_spotcheck(rt, out: Path, args) -> None:
    """Answer-level agreement of the batched decoder with qwen_runtime.generate on a random subset."""
    path = out / 'spotcheck.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    gen = read_jsonl(out / 'gen.jsonl')
    rng = np.random.default_rng(args.seed)
    for k in sorted(rng.choice(len(gen), min(args.spotcheck, len(gen)), replace=False)):
        g = gen[int(k)]
        if g['id'] in finished:
            continue
        c = cases[g['case']]
        prompt = c['prompt'] if g['format'] == 'point' else c['prompt_bbox']
        ref = qr.generate(rt, qr.build(rt, [sr.load_image(c)], prompt), args.max_new_tokens)
        vr = cf.answer_values(rt, rt.tokenizer.encode(ref['text'], add_special_tokens=False), g['format'])
        val = lambda v: None if v is None else [(d['key'], d['value']) for d in v]
        append_jsonl(path, [{'id': g['id'], 'reference': ref['text'], 'batched': g['text'], 'batch_size': g['batch_size'],
                             'text_equal': ref['text'].strip() == g['text'].strip(),
                             'answer_equal': val(vr) == val(g['values'])}])


# ----------------------------------------------------------------------------- mean ablation

def mean_family(group: str) -> str:
    return 'rc' if group == 'rc' else 'mc'


@torch.inference_mode()
def stage_means(rt, out: Path, args) -> None:
    """Mean full-attention head outputs over answer positions of teacher-forced point answers (rc and mc)."""
    cases = load_cases(out)
    gen = [g for g in read_jsonl(out / 'gen.jsonl') if g['format'] == 'point' and g['score']['valid']]
    res = {}
    rng = np.random.default_rng(args.seed)
    for fam in ('rc', 'mc'):
        gs = [g for g in gen if mean_family(g['group']) == fam]
        gs = [gs[i] for i in rng.permutation(len(gs))[:args.n_means]]
        if not gs:
            raise ValueError(f'No valid answers for mean family {fam}')
        acc, n = {l: 0.0 for l in FULL_LAYERS}, 0
        for g in gs:
            c = cases[g['case']]
            prep = cr.prepare(rt, sr.load_image(c), c['prompt'])
            L = len(prep.ids)
            x, pos = cr.teacher_forced(rt, prep, g['ids'][:-1])
            with cfn.TailCapture(rt, L - 1) as cap:
                rt.model(inputs_embeds=x, position_ids=pos, use_cache=False, logits_to_keep=1)
            for l in FULL_LAYERS:
                acc[l] = acc[l] + cap.store[l].float().sum(0).cpu()
            n += cap.store[FULL_LAYERS[0]].shape[0]
        res[fam] = {l: (acc[l] / n).numpy() for l in FULL_LAYERS}
        print(f'means {fam}: {len(gs)} answers, {n} positions', flush=True)
    np.savez_compressed(out / 'means.npz', **{f'{f}_{l}': v for f, d in res.items() for l, v in d.items()})


def head_means(rt, out: Path, fam: str) -> dict[tuple[int, int], torch.Tensor]:
    mz = np.load(out / 'means.npz')
    means = {}
    for l in FULL_LAYERS:
        module, hd = cr.mixer_projection(rt, l)
        m = torch.from_numpy(mz[f'{fam}_{l}']).to(module.weight.device)
        for h in range(16):
            means[(l, h)] = m[h * hd:(h + 1) * hd]
    return means


@torch.inference_mode()
def stage_ablate(rt, out: Path, args) -> None:
    """Greedy point answers with mean-ablated head sets at the answer positions (from the last prompt token)."""
    path = out / 'ablate.jsonl'
    finished = done_ids(path)
    cases = load_cases(out)
    sets = cfn.parse_sets(args.ablate_sets)
    names = list(sets)
    means = {fam: head_means(rt, out, fam) for fam in ('rc', 'mc')}
    for cid, c in cases.items():
        if c['group'] not in args.groups or cid in finished:
            continue
        prep = cr.prepare(rt, sr.load_image(c), c['prompt'])
        L = len(prep.ids)
        x = cr.embeds(rt, prep)
        ops = [(j, l, h, 'mean') for j, n in enumerate(names) for l, h in sets[n]]
        with cr.MixerEdit(rt, ops, L - 1, means=means[mean_family(c['group'])]):
            res = cr.greedy(rt, x.expand(len(names), -1, -1).contiguous(),
                            prep.pos.expand(-1, len(names), -1).contiguous(), args.max_new_tokens)
        rows = []
        for n, r in zip(names, res):
            vals = cf.answer_values(rt, r['ids'], 'point')
            rows.append({'set': n, 'text': r['text'], 'values': vals, 'score': score(c, vals, 'point')})
        append_jsonl(path, [{'id': cid, 'group': c['group'], 'sets': sets, 'results': rows}])
    print('ablate done', flush=True)


# ----------------------------------------------------------------------------- H11 path patching

PATH_LAYERS = (20, 24, 28, 32)


def path_conditions() -> list[tuple[str, list[tuple[int, int]], str]]:
    """(name, [(layer, head)], source) with head = -1 for a whole Gated DeltaNet mixer; source 'B' or 'mean'."""
    dn_pre = [(l, -1) for l in DN_LAYERS if l < 20]
    return [('none', [], 'B'),
            ('H11_B', [(20, 11)], 'B'),
            ('H11_mean', [(20, 11)], 'mean'),
            ('L16H1H2_B', [(16, 1), (16, 2)], 'B'),
            ('L20H3_B', [(20, 3)], 'B'),
            ('L20noH11_B', [(20, h) for h in range(16) if h != 11], 'B'),
            ('L20_B', [(20, h) for h in range(16)], 'B'),
            ('dn_pre20_B', dn_pre, 'B'),
            ('dn_pre20_noH11_B', dn_pre + [(20, 11)], 'mixed')]   # DeltaNet from B but H11 kept from A


@torch.inference_mode()
def stage_path(rt, out: Path, args) -> None:
    """What-pair patching at the answer positions; later-layer attention and first-digit logit differences.

    Run A: original expression, teacher-forced on its own greedy point answer. Run B: the expression of another
    object in the same image, teacher-forced on run A's answer. For each condition the listed component outputs
    of run A at positions >= the last prompt token are replaced by run B's (or by the mean output), and the
    attention rows of layers 20/24/28/32 at both first-digit queries are recorded together with the logits.
    'dn_pre20_noH11_B' patches the DeltaNet mixers of layers 1-19 from run B and additionally pins H11 to
    run A's output, which tests whether the referent change reaches the later heads through H11.
    """
    path = out / 'path.jsonl'
    (out / 'path_attn').mkdir(exist_ok=True)
    cases = load_cases(out)
    gen = {g['id']: g for g in read_jsonl(out / 'gen.jsonl')}
    digits = cf.digit_ids(rt)
    conds = path_conditions()
    means_by_family = {fam: head_means(rt, out, fam) for fam in ('rc', 'mc')}
    todo = [cid for cid, c in cases.items() if c['group'] in args.groups and 'prompt_b' in c
            and f'{cid}:point' in gen and gen[f'{cid}:point']['score']['valid']][:args.n_path]
    for cid in todo:
        if cid in done_ids(path):
            continue
        t0 = time.time()
        c, g = cases[cid], gen[f'{cid}:point']
        means = means_by_family[mean_family(c['group'])]
        img = sr.load_image(c)
        pa, pb = cr.prepare(rt, img, c['prompt']), cr.prepare(rt, img, c['prompt_b'])
        La, Lb = len(pa.ids), len(pb.ids)
        ans = g['ids']
        xa, pos_a = cr.teacher_forced(rt, pa, ans[:-1])
        xb, pos_b = cr.teacher_forced(rt, pb, ans[:-1])
        qa, qb = cf.first_digit_queries(g['values'], La), cf.first_digit_queries(g['values'], Lb)
        with cfn.TailCapture(rt, Lb - 1) as cap:
            lb, att_b, _ = forward_with_attention(rt, xb, pos_b, qb, PATH_LAYERS, Lb - 1)
        donor = cap.store
        R = len(conds)
        x = xa.expand(R, -1, -1).contiguous()
        p = pos_a.expand(-1, R, -1).contiguous()
        ops_b = [(j, l, h) for j, (_, heads, src) in enumerate(conds) if src == 'B' for l, h in heads]
        ops_b += [(j, l, h) for j, (_, heads, src) in enumerate(conds) if src == 'mixed' for l, h in heads if h < 0]
        ops_m = [(j, l, h, 'mean') for j, (_, heads, src) in enumerate(conds) if src == 'mean' for l, h in heads]
        # 'mixed' rows pin H11 to the output it has in the unpatched run A, captured in a separate forward.
        with cfn.TailCapture(rt, La - 1) as cap_a:
            rt.model(inputs_embeds=xa, position_ids=pos_a, use_cache=False, logits_to_keep=1)
        donor_a = {20: cap_a.store[20]}
        ops_a = [(j, 20, 11) for j, (_, heads, src) in enumerate(conds) if src == 'mixed']
        with cfn.TailPatch(rt, donor, ops_b, La - 1), cr.MixerEdit(rt, ops_m, La - 1, means=means), \
                cfn.TailPatch(rt, donor_a, ops_a, La - 1):
            lg, att, chk = forward_with_attention(rt, x, p, qa, PATH_LAYERS, La - 1)
        lds = cfn.value_lds(lb[0].to(lg.device), lg, ans, g['values'], digits)
        sa, ea = pa.span['start'], pa.span['end']
        sb, eb = pb.span['start'], pb.span['end']
        A = np.stack([att[l][:, :, :, sa:ea].numpy() for l in PATH_LAYERS], 2)       # (R, nq, 4, 16, Nv)
        Bm = np.stack([att_b[l][0, :, :, sb:eb].numpy() for l in PATH_LAYERS], 1)    # (nq, 4, 16, Nv)
        np.savez_compressed(out / 'path_attn' / f'{cid}.npz', a=A.astype(np.float16), b=Bm.astype(np.float16))
        append_jsonl(path, [{'id': cid, 'group': c['group'], 'grid': list(pa.grid), 'conds': [n for n, _, _ in conds],
                             'values': [{k: d[k] for k in ('key', 'axis', 'dA', 'dB', 'LD_B', 'ld')} for d in lds],
                             'attn_check': chk, 'seconds': round(time.time() - t0, 1)}])
        print(f'path {cid} {time.time() - t0:.1f}s', flush=True)
        del donor, donor_a


# ----------------------------------------------------------------------------- main

STAGES = {'gen': stage_gen, 'spotcheck': stage_spotcheck, 'means': stage_means, 'ablate': stage_ablate,
          'path': stage_path}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stages', nargs='+', required=True, choices=['cases', *STAGES])
    p.add_argument('--model', default='Qwen3.5-9B')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--round2', type=Path, required=True, help='3.11 round-2 run directory (RefCOCO cases)')
    p.add_argument('--coco-dir', type=Path, help='directory holding COCO train2017/')
    p.add_argument('--paco-dir', type=Path, help='PACO-LVIS annotations directory')
    p.add_argument('--voc-dir', type=Path, help='VOC2010 directory (Annotations, JPEGImages, SegmentationObject)')
    p.add_argument('--vocpart-json', type=Path, help='Pascal-Part annotations in COCO format')
    p.add_argument('--stuff-dir', type=Path, help='COCO-Stuff train2017 label maps')
    p.add_argument('--per-category-paco', type=int, default=4)
    p.add_argument('--per-category-voc', type=int, default=14)
    p.add_argument('--per-category-stuff', type=int, default=6)
    p.add_argument('--stuff-scan-limit', type=int, default=20000)
    p.add_argument('--groups', nargs='+', default=['rc', 'paco', 'vocpart', 'stuff'])
    p.add_argument('--ablate-sets', default='')
    p.add_argument('--n-means', type=int, default=64)
    p.add_argument('--n-path', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--prep-chunk', type=int, default=64)
    p.add_argument('--max-new-tokens', type=int, default=48)
    p.add_argument('--spotcheck', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--max-memory', default='')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = list(args.stages)
    if 'cases' in stages:
        stage_cases(args.output, args)
        stages.remove('cases')
    if not stages:
        return
    rt = cf.load_runtime(args.model, args.max_memory)
    for st in stages:
        meta = {'stage': st, 'args': {k: str(v) for k, v in vars(args).items()}, 'runtime': qr.describe(rt),
                'code_sha256': sr.sha256(Path(__file__)), 'started': datetime.now(timezone.utc).isoformat(),
                'device': torch.cuda.get_device_name(0)}
        append_jsonl(args.output / 'run_log.jsonl', [meta])
        STAGES[st](rt, args.output, args)


if __name__ == '__main__':
    main()
