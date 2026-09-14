"""Propose ROI trajectories with an offline VLM; proposals require human review."""
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
from pathlib import Path

from data.sft.core.annotations import source_key


def parse_proposal(response: str, width: int, height: int) -> list[dict]:
    text = response.strip()
    if text.startswith('```'):
        text = text.split('\n',1)[1].rsplit('```',1)[0].strip()
    proposal = json.loads(text)
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
    p.add_argument('--gpu',default='5')
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
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
    from eva.inference import resolve_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for local teacher')
    set_seed(0)
    model_path=resolve_model(args.model,args.cache_root)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa').eval()
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
            chat=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
            text=processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True, **({'enable_thinking': False} if model.config.model_type == 'qwen3_5' else {}))
            inputs=processor(text=[text],images=[image],return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            tokens=generated[0,inputs['input_ids'].shape[1]:]
            response=processor.decode(tokens,skip_special_tokens=True)
            key=source_key(base['source_id'],base['query'],base['answer'],base['image'])
            raw={'source_key':key,'source_id':base['source_id'],'response':response,'model_path':str(model_path),'prompt':prompt,'generated_tokens':len(tokens)}
            eos=model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            try:
                if not any(token in eos for token in tokens.tolist()):
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
