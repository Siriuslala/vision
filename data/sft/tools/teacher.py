"""Export supplemental ROI proposals; automatic.py provides generation plus checks."""
from __future__ import annotations

import argparse
import importlib
import io
import json
from pathlib import Path

from data.sft.core.annotations import source_key
from data.sft.tools.local_teacher import LocalTeacher, json_object


def parse_proposal(response: str, width: int, height: int) -> list[dict]:
    proposal = json_object(response)
    events = proposal['events']
    if not isinstance(events,list) or not events or len(events)>24:
        raise ValueError('Expected 1..24 events')
    from data.sft.core.common import pixel_box
    count = 0
    result = []
    for event in events:
        if event['type'] == 'visual_read':
            # Teacher coordinates use a separately declared [0,1000] scale per axis.
            box = event['box']
            if len(box)!=4 or any(type(x) not in (int,float) for x in box):
                raise ValueError('Invalid teacher box')
            pixel = [box[0]*width/1000,box[1]*height/1000,box[2]*width/1000,box[3]*height/1000]
            pixel_box(pixel,width,height)
            count += 1
            result.append({'type':'visual_read','box':pixel,'reference_region_id':0})
        elif event['type'] == 'reasoning_text':
            if not isinstance(event.get('text'),str) or not event['text'].strip():
                raise ValueError('Empty reasoning text')
            result.append({'type':'reasoning_text','text':event['text']})
        else:
            raise ValueError('Unknown proposal event')
    if count > 4:
        raise ValueError('Teacher exceeded four-read budget')
    if not count:
        raise ValueError('No visual observations')
    if not any(event['type'] == 'reasoning_text' for event in result):
        raise ValueError('Interleaved teacher proposal lacks natural-language reasoning')
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['cosyn','llava'],required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='Qwen3.5-9B')
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--gpu',default='4')
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--max-new-tokens',type=int,default=1536)
    args=p.parse_args()
    if args.limit<1:
        p.error('Teacher runs require a positive explicit limit')
    if args.max_new_tokens < 1:
        p.error('max-new-tokens must be positive')
    if args.output.resolve().is_relative_to(args.source_root.resolve()):
        p.error('Teacher outputs must be outside source downloads')
    if args.output.exists() or args.output.with_suffix('.raw.jsonl').exists():
        p.error('Output exists')
    from PIL import Image
    teacher = LocalTeacher(args.model,args.cache_root,args.gpu)
    model_path = teacher.info['snapshot']
    adapter=importlib.import_module('data.sft.datasets.'+args.dataset)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    proposed_count = 0
    with args.output.open('x') as output, args.output.with_suffix('.raw.jsonl').open('x') as raw_output:
        for base in adapter.iter_tasks(args.source_root,args.limit):
            if 'reject_reason' in base:
                raw_output.write(json.dumps(base,ensure_ascii=False)+'\n')
                continue
            image=Image.open(io.BytesIO(base['image'])).convert('RGB')
            prompt=('Propose a short visual observation sequence to answer the question from this image. '
                    'Return ONLY a JSON object with an events array. Each event is either '
                    '{"type":"visual_read","box":[left,top,right,bottom]} or '
                    '{"type":"reasoning_text","text":"..."}. '
                    'Box coordinates use 0..1000 separately along image width and height. '
                    'Select actual evidence regions, keeping labels and context needed for interpreting the evidence. '
                    'After EVERY visual_read event, include a reasoning_text event stating the specific visible finding and how it helps answer the question. An events array containing only boxes is invalid. Use only visible facts. '
                    'Do not claim access to hidden data, code or a solver. Use at most 4 reads. '
                    'Do not output a finish event or final answer field. Question: '+base['query'])
            # The source answer is intentionally withheld from proposal generation.
            result = teacher.generate([image],prompt,args.max_new_tokens)
            response = result['response']
            key=source_key(base['source_id'],base['query'],base['answer'],base['image'])
            raw={**result,'source_key':key,'source_id':base['source_id'],'model_path':str(model_path)}
            try:
                if result['finish_reason']!='eos':
                    raise ValueError('Teacher output truncated without EOS')
                events=parse_proposal(response,*image.size)
                annotation={'source_key':key,'status':'proposed','reviewer':None,'events':events,'visual_only_safe':False,'visual_only_basis':'','teacher':{'model':str(model_path),'seed':0,'max_new_tokens':args.max_new_tokens},'source_answer_withheld':True}
                output.write(json.dumps(annotation,ensure_ascii=False,allow_nan=False)+'\n')
                output.flush()
                raw['status']='proposed'
                proposed_count += 1
            except (ValueError,KeyError,TypeError) as exc:
                raw.update(status='rejected',error=str(exc))
            raw_output.write(json.dumps(raw,ensure_ascii=False)+'\n')
            raw_output.flush()
            print(base['source_id'],raw['status'],flush=True)

    if not proposed_count:
        raise RuntimeError('No usable teacher proposals; inspect raw rejection log')


if __name__=='__main__':
    main()
