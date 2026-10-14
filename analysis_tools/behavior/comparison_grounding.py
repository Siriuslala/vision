"""Factor target selection and output geometry, including reversed majority features."""
import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq

from analysis_tools.behavior.grounding_pilot import make_grid


def majority_image(target: int, minority: str) -> Image.Image:
    """Change feature frequency without revealing the minority attribute to a query."""
    image, _ = make_grid(6, False, divmod(target, 6), 'normal')
    draw = ImageDraw.Draw(image)
    if minority not in ('gap', 'intact'):
        raise ValueError('Unknown minority attribute.')
    for i in range(36):
        if (i == target) == (minority == 'gap'):
            x, y = i % 6 * 64, i // 6 * 64
            draw.rectangle((x+39, y+32, x+46, y+33), fill='white')
    return image


def figure_boxes(image: Image.Image) -> list[list[float]]:
    """Use tight half-open pixel boxes; no learned detector or answer prompt is used."""
    pixels = np.asarray(image)
    boxes = []
    for i in range(36):
        x, y = i % 6 * 64, i // 6 * 64
        ys, xs = np.nonzero(np.any(pixels[y:y+64, x:x+64] != 255, axis=-1))
        if not len(xs):
            raise ValueError('Empty figure cell.')
        boxes.append([float(x+xs.min()), float(y+ys.min()), float(x+xs.max()+1), float(y+ys.max()+1)])
    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / (area_a + area_b - intersection)


def score_answer(case: dict, record: dict, *, allow_bare_coordinates: bool = False) -> dict:
    """Reject incomplete, ambiguous, and malformed answers without oracle extraction."""
    result = dict(valid=False, correct=False)
    if record['finish_reason'] != 'eos':
        return result
    body = re.sub(r'^```(?:json)?\s*|\s*```$', '', record['response'].split('</think>')[-1].strip())
    try:
        value = json.loads(body)
        if case['format'] == 'row_column':
            row, col = value['row'], value['column']
            if type(row) is not int or type(col) is not int or not (1 <= row <= 6 and 1 <= col <= 6):
                return result
            return dict(valid=True, correct=(row-1)*6+col-1 == case['target'])
        key, length = ('point_2d', 2) if case['format'] == 'point' else ('bbox_2d', 4)
        if allow_bare_coordinates and isinstance(value, list) and len(value) == length and all(type(v) in (float,int) for v in value):
            value = [{key: value}]
        if not isinstance(value, list) or len(value) != 1:
            return result
        coordinates = value[0][key]
        if not isinstance(coordinates, list) or len(coordinates) != length:
            return result
        if any(type(v) not in (float,int) or not np.isfinite(v) or not 0 <= v <= 1000 for v in coordinates):
            return result
    except (ValueError, TypeError, KeyError):
        return result
    size = np.array(case['size'])
    pixel = np.array(coordinates) * np.tile(size, length//2) / 1000
    box = np.array(case['bbox'])
    if length == 4:
        if pixel[0] >= pixel[2] or pixel[1] >= pixel[3]:
            return result
        iou = box_iou(pixel.tolist(), box.tolist())
        result = dict(valid=True, correct=iou >= .5, iou=iou, pixel=pixel.tolist())
        if case['source'] == 'synthetic':
            centers = np.array([(np.array(b[:2])+np.array(b[2:]))/2 for b in case['boxes']])
            distances = np.linalg.norm(centers-(pixel[:2]+pixel[2:])/2,axis=1)
            unique = np.isclose(distances,distances.min(),rtol=0,atol=1e-6).sum()==1
            result['identity_correct'] = bool(unique and distances.argmin()==case['target'])
        return result
    center = (box[:2]+box[2:])/2
    error = float(np.linalg.norm(pixel-center))
    diagonal = float(np.linalg.norm(box[2:]-box[:2]))
    result = dict(valid=True, correct=error <= .1*diagonal, error_pixels=error,
                  error_bbox_diagonal=error/diagonal, pixel=pixel.tolist(),
                  point_in_box=bool(np.all(pixel >= box[:2]) and np.all(pixel < box[2:])))
    if case['source'] == 'synthetic':
        centers = np.array([(np.array(b[:2])+np.array(b[2:]))/2 for b in case['boxes']])
        distances = np.linalg.norm(centers-pixel,axis=1)
        unique = np.isclose(distances, distances.min(), rtol=0, atol=1e-6).sum() == 1
        result['identity_correct'] = bool(unique and distances.argmin() == case['target'])
    return result


def build_cases(output: Path, refcoco: Path, natural_count: int) -> list[dict]:
    suffixes = {
        'point': ' Give the center of its bounding box. Return only [{"point_2d": [x, y]}].',
        'bbox': ' Give its tight bounding box. Return only [{"bbox_2d": [x_min, y_min, x_max, y_max]}].',
        'row_column': ' Return only {"row": r, "column": c}, counting from top to bottom and left to right starting at 1.'}
    cases = []
    def add(image, stem, query, task, fmt, **metadata):
        path = output / f'{stem}.png'
        image.save(path)
        prompt = query + suffixes[fmt]
        if fmt != 'row_column':
            prompt += ' All coordinates use 0-1000 normalization; x increases rightward and y downward. Do not explain.'
        cases.append(dict(id=f'{stem}_{task}_{fmt}',image=str(path),size=image.size,
                          prompt=prompt,task=task,format=fmt,image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**metadata))
    for target in (8,21,28):
        r,c=divmod(target,6)
        for minority in ('gap','intact','red'):
            image = make_grid(6,False,(r,c),'red')[0] if minority == 'red' else majority_image(target,minority)
            boxes = figure_boxes(image)
            attributes = dict(gap='Find the one person whose arm on the right side of the image has a gap.',
                              intact='Find the one person whose arm on the right side of the image is complete, without a gap.',
                              red='Find the red person.')
            queries = [('attribute',attributes[minority])]
            if minority != 'red':
                queries += [('comparison','One person differs from all the others. Find that person.'),
                            ('ordinal',f'Find the person in row {r+1}, column {c+1}, counting from top to bottom and left to right starting at 1.')]
            for task,query in queries:
                for fmt in suffixes:
                    if task == 'ordinal' and fmt == 'row_column':
                        continue
                    add(image,f'{minority}_target{target}',query,task,fmt,source='synthetic',minority=minority,
                        target=target,boxes=boxes,bbox=boxes[target])
    seen=set()
    if natural_count:
        shards=sorted((refcoco/'data').glob('val-*.parquet'))
        if not shards:
            raise FileNotFoundError('Missing RefCOCO val shards.')
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    raw=row['image']['bytes']; digest=hashlib.sha256(raw).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    image=Image.open(BytesIO(raw)).convert('RGB')
                    x,y,w,h=row['bbox']; box=[x,y,x+w,y+h]
                    if w<=0 or h<=0 or x<0 or y<0 or x+w>image.width+.01 or y+h>image.height+.01:
                        raise ValueError('Invalid RefCOCO bbox.')
                    for fmt in ('point','bbox'):
                        add(image,f"refcoco_{row['question_id']}",f"Locate the object described as: {row['answer'][0]}",
                            'referring_expression',fmt,source='refcoco',minority=None,bbox=box,question_id=row['question_id'],
                            referring_expression=row['answer'][0],source_image_sha256=digest)
                    if len(seen)>=natural_count:
                        return cases
        raise ValueError('Insufficient distinct RefCOCO images.')
    return cases


def score_predictions(output: Path, *, geometry_only: bool = False) -> None:
    """Preserve strict scores; optionally report unambiguous bare coordinates separately."""
    cases=json.loads((output/'cases.json').read_text())
    predictions=[json.loads(s) for s in (output/'predictions.jsonl').read_text().splitlines()]
    groups=defaultdict(list); records=[]
    for case,record in zip(cases,predictions,strict=True):
        if case['prompt']!=record['prompt'] or case['image']!=record['image_path']:
            raise ValueError('Mismatched input and prediction.')
        score=score_answer(case,record,allow_bare_coordinates=geometry_only)
        records.append(dict(id=case['id'],**score))
        groups[f"{case['source']}/{case['minority']}/{case['task']}/{case['format']}"].append(score)
    summary={k:dict(total=len(v),valid=sum(x['valid'] for x in v),correct=sum(x['correct'] for x in v),
                    identity_correct=sum(x.get('identity_correct',False) for x in v)) for k,v in groups.items()}
    name='scores_geometry.json' if geometry_only else 'scores.json'
    payload=dict(summary=summary,records=records,allow_bare_coordinates=geometry_only,
                 code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/name).write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refcoco',type=Path,required=True)
    p.add_argument('--natural-count',type=int,default=8)
    p.add_argument('--model',default='Qwen3.5-9B')
    p.add_argument('--gpu-ids',default='7')
    p.add_argument('--max-new-tokens',type=int,default=512)
    p.add_argument('--score-geometry-only',action='store_true')
    args=p.parse_args()
    if args.natural_count<0 or args.max_new_tokens<1:
        p.error('Invalid count or token budget.')
    if args.score_geometry_only:
        if (args.output/'scores_geometry.json').exists():
            raise FileExistsError('Geometry diagnostic already exists.')
        score_predictions(args.output,geometry_only=True)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    cases=build_cases(args.output,args.refcoco,args.natural_count)
    (args.output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (args.output/'experiment.json').write_text(json.dumps(dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         point_tolerance_bbox_diagonal=.1,bbox_iou_threshold=.5,thinking='off',source_bbox_convention='half_open_xyxy'),indent=2)+'\n')
    output=args.output/'predictions.jsonl'
    sys.argv=['comparison_grounding','--model',args.model,'--gpu-ids',args.gpu_ids,'--output',str(output),
              '--thinking','off','--batchsize','1','--max-new-tokens',str(args.max_new_tokens),
              '--image-path',*[c['image'] for c in cases],'--prompt',*[c['prompt'] for c in cases]]
    from eva.inference import main as infer
    infer()
    score_predictions(args.output)
    score_predictions(args.output,geometry_only=True)


if __name__=='__main__':
    main()
