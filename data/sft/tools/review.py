"""Render bounded review sheets and admit explicitly reviewed candidates."""
from __future__ import annotations
import argparse
import copy
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw
from data.sft.core.common import digest, validate_record


def render(records_path: Path, data_root: Path, output: Path, limit: int) -> None:
    output.mkdir(parents=True,exist_ok=False)
    sections = []
    decisions = []
    with records_path.open() as source:
        for index, line in enumerate(source):
            if limit and index >= limit:
                break
            record=json.loads(line)
            validate_record(record,data_root)
            picture=Image.open(data_root/record['image']['path']).convert('RGB')
            scale=max(picture.size)
            draw=ImageDraw.Draw(picture)
            for event in record['trajectory']:
                if event['type']!='visual_read':
                    continue
                cx,cy=event['region']['center'];w,h=event['region']['size']
                box=[(cx-w/2)*scale,(cy-h/2)*scale,(cx+w/2)*scale,(cy+h/2)*scale]
                draw.rectangle(box,outline='red',width=max(2,scale//300))
                draw.text((box[0]+3,box[1]+3),str(event['region_id']),fill='red')
            name=f'{index:04d}.jpg'
            picture.thumbnail((1400,1000));picture.save(output/name)
            esc=lambda v:html.escape(json.dumps(v,ensure_ascii=False,indent=2))
            sections.append(f'<article><h2>{html.escape(record["sample_id"])}</h2><p>{html.escape(record["query"])}</p><img src="{name}"><h3>Answer</h3><pre>{esc(record["answer"])}</pre><h3>Events</h3><pre>{esc(record["trajectory"])}</pre><details><summary>Provenance and offline metadata</summary><pre>{esc(record["metadata"])}</pre></details></article>')
            decisions.append({'sample_id':record['sample_id'],'candidate_sha256':digest(record),'decision':'pending','reviewer':'','task_and_answer_checked':False,'observation_sequence_checked':False,'feature_readability_checked':False,'notes':''})
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SFT sample review</title><style>body{max-width:1100px;margin:auto;font:16px sans-serif}img{max-width:100%}pre{white-space:pre-wrap}article{border-bottom:2px solid #bbb;padding:24px}</style>'+''.join(sections))
    (output/'decisions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in decisions))


def admit(records_path: Path, data_root: Path, decisions_path: Path, batch_path: Path, output: Path) -> None:
    import hashlib
    batch=json.loads(batch_path.read_text())
    if batch.get('candidates_sha256') != hashlib.sha256(records_path.read_bytes()).hexdigest():
        raise ValueError('Batch approval does not match candidate file')
    for field in ('reviewer','report','source_overlap_check','feature_config_id'):
        if not isinstance(batch.get(field),str) or not batch[field].strip():
            raise ValueError(f'Batch approval requires {field}')
    if batch.get('approved') is not True:
        raise ValueError('Batch quality approval is required')
    decisions={}
    for line in decisions_path.read_text().splitlines():
        row=json.loads(line)
        if row['sample_id'] in decisions:
            raise ValueError('Duplicate review decision')
        decisions[row['sample_id']]=row
    accepted=[]
    image_splits={}
    group_splits={}
    for line in records_path.read_text().splitlines():
        record=json.loads(line)
        for registry,key in ((image_splits,record['image']['sha256']),(group_splits,record['provenance']['group_id'])):
            split=record['provenance']['split']
            if key in registry and registry[key]!=split:raise ValueError('Split conflict in reviewed batch')
            registry[key]=split
        decision=decisions.get(record['sample_id'])
        if not decision or decision['decision']!='accept':
            continue
        if decision['candidate_sha256']!=digest(record):
            raise ValueError('Candidate changed after review')
        if not decision['reviewer'] or not all(decision.get(f) is True for f in ('task_and_answer_checked','observation_sequence_checked','feature_readability_checked')):
            raise ValueError('Incomplete sample review')
        validate_record(record,data_root)
        record=copy.deepcopy(record)
        record['quality']={'status':'validated','notes':decision['notes']}
        record['metadata']['acceptance']={'sample_review':decision,'batch_review':batch}
        accepted.append(record)
    if not accepted:
        raise ValueError('No explicitly accepted records')
    from data.sft.core.splits import register
    registry_path=Path(batch['split_registry'])
    if not registry_path.is_file():raise ValueError('Shared split registry is missing')
    register(accepted,registry_path)
    with output.open('x') as f:
        for record in accepted:
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['render','admit'])
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=5)
    p.add_argument('--decisions',type=Path)
    p.add_argument('--batch-review',type=Path)
    a=p.parse_args()
    if a.limit<0:p.error('limit must be nonnegative')
    if a.stage=='render':render(a.records,a.data_root,a.output,a.limit)
    else:
        if not a.decisions or not a.batch_review:p.error('admit requires decisions and batch-review')
        admit(a.records,a.data_root,a.decisions,a.batch_review,a.output)

if __name__=='__main__':main()
