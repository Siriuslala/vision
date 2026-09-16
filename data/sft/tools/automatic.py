"""Generate and check visual SFT trajectories before final human inspection."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import re

from PIL import Image

from data.sft.core.common import build_records, pixel_box
from data.sft.core.splits import register
from data.sft.tools.local_teacher import LocalTeacher, json_object
from data.sft.tools.window_catalog import candidate_windows, catalog_prompt, contact_sheet
from data.sft.datasets.llava import requires_instance_annotations


def normalized_answer(value) -> str:
    return re.sub(r'\s+', ' ', str(value).strip().strip('"\'').rstrip('.')).casefold()


def parse_generation(result: dict, width: int, height: int, max_reads: int = 6, windows: list | None = None) -> list[dict]:
    if type(result.get('answerable')) is not bool or not result['answerable']:
        raise ValueError('Teacher could not answer from the image')
    if not isinstance(result.get('answer'), str) or not result['answer'].strip():
        raise ValueError('Missing independent teacher answer')
    if windows is not None and 'steps' in result:
        steps = result['steps']
        if not isinstance(steps,list) or len(steps)>max_reads:
            raise ValueError('Invalid catalog steps')
        events = []
        previous_window = None
        for step in steps:
            window_id = step['window_id']
            if type(window_id) is not int or not 1<=window_id<=len(windows):
                raise ValueError('Teacher selected an unknown window')
            # Consecutive identical reads expose exactly the same pixels. Keep all
            # findings after the first read; revisits after other windows remain.
            if window_id!=previous_window:
                events.append({'type':'visual_read','window_id':window_id})
            events.append({'type':'reasoning_text','text':step['finding']})
            previous_window = window_id
        if steps:
            events.append({'type':'reasoning_text','text':result.get('conclusion','')})
    else:
        events = result.get('events')
    if not isinstance(events, list) or len(events) > 32:
        raise ValueError('Invalid teacher event list')
    converted, reads = [], 0
    window_to_read = {}
    for event in events:
        if event['type'] == 'visual_read':
            if windows is not None:
                window_id = event.get('window_id')
                if type(window_id) is not int or not 1 <= window_id <= len(windows):
                    raise ValueError('Teacher selected an unknown window')
                box = list(windows[window_id-1])
            else:
                box = event['box']
            if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int, float) for x in box):
                raise ValueError('Invalid teacher box')
            if windows is None:
                box = [box[0]*width/1000, box[1]*height/1000, box[2]*width/1000, box[3]*height/1000]
            pixel_box(box, width, height)
            ref = event.get('reference_region_id', 0)
            if type(ref) is not int or not 0 <= ref <= reads:
                raise ValueError('Invalid teacher reference')
            # References to old windows must remain executable under the model transform.
            if ref:
                parent = [x for x in converted if x['type']=='visual_read'][ref-1]['box']
                center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
                pc = [(parent[0]+parent[2])/2, (parent[1]+parent[3])/2]
                size = [parent[2]-parent[0], parent[3]-parent[1]]
                if any(abs(a-b) >= s for a,b,s in zip(center,pc,size)) or box[2]-box[0] >= 2*size[0] or box[3]-box[1] >= 2*size[1]:
                    raise ValueError('Teacher reference cannot execute the requested move/scale')
            converted.append({'type':'visual_read','box':box,'reference_region_id':ref})
            reads += 1
            if windows is not None:
                window_to_read[window_id] = reads
        elif event['type'] == 'reasoning_text':
            if not isinstance(event.get('text'), str) or not event['text'].strip():
                raise ValueError('Empty teacher reasoning')
            if windows is not None:
                for match in re.finditer(r'\bwindows?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)',event['text'],re.I):
                    if any(window_to_read.get(int(value))!=int(value) for value in re.findall(r'\d+',match[1])):
                        raise ValueError('Reasoning uses temporary proposal IDs instead of actual observation IDs')
            converted.append({'type':'reasoning_text','text':event['text']})
        else:
            raise ValueError('Unknown teacher event')
    if reads > max_reads:
        raise ValueError('Teacher exceeded reading budget')
    if reads and not any(e['type']=='reasoning_text' for e in converted):
        raise ValueError('Interleaved trajectory has no reasoning')
    if not reads and events:
        raise ValueError('Direct-answer examples must have an empty intermediate sequence')
    return converted


def curriculum(events: list[dict], variant: str) -> dict:
    reads = sum(event['type']=='visual_read' for event in events)
    return {'trajectory_type': variant, 'read_count': reads, 'read_group': 'zero_read' if reads==0 else 'one_read' if reads==1 else 'multi_read'}


def verify_verdict(verdict: dict, source_answer: dict, generated_answer: str, events: list[dict], variant: str, semantic_agreement: bool = False) -> None:
    fields = ('answer_correct','observations_supported','reasoning_complete','reads_useful','self_correction')
    if variant=='normal':
        fields += ('visual_only_safe',)
    if any(type(verdict.get(key)) is not bool for key in fields):
        raise ValueError('Incomplete teacher quality verdict')
    if not isinstance(verdict.get('reason'), str) or not verdict['reason'].strip():
        raise ValueError('Quality verdict must explain its evidence')
    if not all(verdict[k] for k in ('answer_correct','observations_supported','reasoning_complete','reads_useful')):
        raise ValueError('Teacher quality rejection: '+verdict['reason'])
    if variant == 'normal' and verdict['self_correction']:
        raise ValueError('Normal trajectory contains an incorrect step followed by correction')
    if variant == 'normal' and verdict.get('error_type','none')!='none':
        raise ValueError('Normal trajectory has a contradictory error classification')
    if variant == 'self_correction' and not verdict['self_correction']:
        raise ValueError('Source trajectory was not confirmed as self-correction')
    if variant == 'self_correction' and verdict.get('error_type') not in ('wrong_region','wrong_fact'):
        raise ValueError('Self-correction requires a wrong target region or factual error, not merely adding context')
    if not semantic_agreement and normalized_answer(generated_answer) != normalized_answer(source_answer['value']):
        raise ValueError('Independent teacher answer disagrees with source answer')


def collect_tasks(dataset: str, root: Path, scan_limit: int, limit: int, subsets: list[str], per_image: int = 2, per_shard_limit: int = 5, skip_images: int = 0, source_ids: list[str] | None = None):
    adapter = importlib.import_module('data.sft.datasets.'+dataset)
    tasks, failures, seen = [], [], set()
    counts = Counter()
    skipped_images = set()
    fallback = []
    roots = [root / subset for subset in subsets] if subsets else [root]
    # Round-robin across requested domains avoids using only the first sorted shard.
    options = {'per_shard_limit': per_shard_limit} if dataset == 'thyme' else {}
    streams = [iter(adapter.iter_tasks(path, scan_limit, **options)) for path in roots]
    active = list(range(len(streams)))
    while active and len(tasks)<limit:
        for index in list(active):
            try:
                base = next(streams[index])
            except StopIteration:
                active.remove(index); continue
            if 'reject_reason' in base:
                failures.append(base); continue
            if source_ids and base['source_id'] not in source_ids:
                continue
            if requires_instance_annotations(base['query'],base.get('metadata',{}).get('source_subset',''),base['task_type']):
                failures.append({'source_id':base['source_id'],'reject_reason':'Direct image counting requires instance annotations and counting metadata'})
                continue
            raw = base['image'] if isinstance(base['image'],bytes) else Path(base['image']).read_bytes()
            base['image'] = raw
            sha = hashlib.sha256(raw).hexdigest()
            if sha in skipped_images:
                continue
            if len(skipped_images)<skip_images:
                skipped_images.add(sha)
                continue
            if base['source_id'] in seen or counts[sha]>=per_image:
                continue
            # Keep a small number of single-read controls while prioritizing comparisons.
            text = base['query'].lower()
            score = int(any(w in text for w in ('compare','both','between','more','less','than','total','difference','behind','right','left','above','below','cost','enough','which')))
            if not source_ids and not score and len(tasks)>=max(1,limit//4):
                if len(fallback)<limit:
                    fallback.append((base,sha))
                continue
            counts[sha]+=1;seen.add(base['source_id']);tasks.append(base)
            if len(tasks)>=limit: break
    for base,sha in fallback:
        if len(tasks)>=limit:
            break
        if base['source_id'] not in seen and counts[sha]<per_image:
            counts[sha]+=1;seen.add(base['source_id']);tasks.append(base)
    return tasks, failures


def generation_prompt(query: str) -> str:
    return ('Solve the question using only this image. First decide which image details must be read. '
      'Produce a correct reasoning sequence with no intentional mistakes or self-correction. '
      'For comparisons, read the relevant objects, labels, values or options in meaningful order. '
      'Use two or more distinct observations only when they provide different necessary evidence. '
      'Do not duplicate windows just to increase the count. At most six observations. '
      'A direct answer may have events=[] if no focused observation is useful. '
      'After observations, write clear natural reasoning including required calculations and the conclusion. '
      'Return ONLY JSON: {"answerable":true,"answer":"short final answer exactly as requested (option letter for multiple choice)",'
      '"events":[{"type":"visual_read","box":[left,top,right,bottom],"reference_region_id":0},'
      '{"type":"reasoning_text","text":"specific finding and its role in the solution"}]}. '
      'Box coordinates are 0..1000 separately for image width and height. Include entire relevant objects/text plus context. '
      'Reference 0 means original image. Never put the source answer or hidden data in the reasoning. Question: '+query)


def check_prompt(base: dict, answer: str, events: list[dict], variant: str) -> str:
    return ('Act as a strict visual dataset checker. The first image is the original. Subsequent images are the actual '
      'cropped observation windows, in reading order. Check every factual statement against the original and the available windows. '
      'A statement may use earlier observations; do not assume hidden numeric tables or object labels. '
      'Check the solution independently, missing evidence, incorrect object colors/counts, incorrect arithmetic and redundant reads. '
      'Self-correction means an actual mistaken belief/selection is explicitly recognized and corrected, not simply observing another region. '
      'Reading a correct anchor object and then adding another relevant object or wider context is normal reasoning, even if the source text calls it a correction. '
      'For self-correction, early acknowledged wrong beliefs are allowed, but the correction must be supported by later observations. '
      'visual_only_safe asks whether all necessary input evidence remains in the question and observations after removing text. '
      'Arithmetic or logical deductions from that evidence do not require preserving text; only extra indispensable information carried solely in text makes it unsafe. '
      'Never claim that latent reasoning capability has been tested. '
      'Return ONLY JSON with boolean answer_correct, observations_supported, reasoning_complete, reads_useful, '
      'self_correction, visual_only_safe, error_type (none, wrong_region, or wrong_fact), and a specific string reason. '
      'When self_correction is true, error_type must name the actual wrong_region or wrong_fact. '
      'When it is false, error_type must be none. These fields must agree. '
      'Question: '+base['query']+'\nProposed final answer: '+answer+'\nTrajectory type: '+variant+
      '\nEvents (pixel coordinates): '+json.dumps(events,ensure_ascii=False))


def process_task(base: dict, dataset: str, teacher, run: Path, data_root: Path, max_tokens: int, raw_log, include_source: bool = True, window_method: str = 'coordinates', crop_check: bool = True, retry_feedback: dict | None = None):
    image = Image.open(io.BytesIO(base['image'])).convert('RGB')
    identity = base['source_id']
    attempts = []
    independent_answer = None
    agreement = {'method':'normalized_exact','equivalent':False}
    source_boxes = base.get('source_boxes') or [e['box'] for e in base.get('source_events',[]) if e['type']=='visual_read']
    windows, method = candidate_windows(image, source_boxes, table=base['task_type']=='table_qa',chart=base['task_type']=='chart_qa') if window_method=='catalog' else (None, 'teacher_coordinates')
    if dataset=='visual_cot' and window_method=='catalog' and base.get('source_boxes'):
        windows, method = base['source_boxes'], 'source_annotated_regions'
    generation_images = [image,contact_sheet(image,windows,large=base['task_type']=='chart_qa')] if windows is not None else [image]
    prompt = catalog_prompt(base['query'], windows, method) if windows is not None else generation_prompt(base['query'])
    if window_method=='direct':
        method = 'direct_answer_control'
        prompt = ('Answer this question directly from the original image, without any additional observation or intermediate reasoning events. '
            'Return ONLY JSON {"answerable":true,"answer":"short final answer, with an option letter if requested","events":[]}. '
            'If the visible image does not support a confident answer, set answerable=false. The events list must be empty. Question: '+base['query'])
    if retry_feedback:
        prompt += '\nA previous proposal failed checks. Reconsider the image and select sufficient, nonredundant windows. The source answer remains withheld. Diagnostic feedback: '+json.dumps(retry_feedback,ensure_ascii=False)
    generation = teacher.generate(generation_images, prompt, max_tokens)
    generation.update(source_id=identity, stage='generation')
    raw_log.write(json.dumps(generation,ensure_ascii=False)+'\n');raw_log.flush()
    try:
        if generation['finish_reason']!='eos': raise ValueError('Generation truncated')
        result = json_object(generation['response'])
        events = parse_generation(result,*image.size, windows=windows)
        if window_method=='direct' and events:
            raise ValueError('Direct-answer control produced intermediate events')
        agreement['equivalent'] = normalized_answer(result['answer'])==normalized_answer(base['answer']['value'])
        if not agreement['equivalent'] and base['answer']['type']=='text':
            comparison = teacher.generate([], 'Compare two answers to the same question for strict semantic equivalence. '
                'A short yes/no may equal a full sentence restating the question. Ignore harmless wording differences. '
                'Reject different numbers, objects, attributes, choices, or additional substantive claims not entailed by the other answer and question. '
                'Do not solve the question or prefer either answer. Return ONLY JSON {"equivalent":true,"reason":"explanation"}. '
                'Question: '+base['query']+'\nAnswer A: '+result['answer']+'\nAnswer B: '+str(base['answer']['value']),max_tokens)
            comparison.update(source_id=identity,stage='answer_equivalence')
            raw_log.write(json.dumps(comparison,ensure_ascii=False)+'\n');raw_log.flush()
            if comparison['finish_reason']!='eos':raise ValueError('Answer comparison truncated')
            agreement = {**json_object(comparison['response']),'method':'teacher_semantic_equivalence'}
            if type(agreement.get('equivalent')) is not bool or not isinstance(agreement.get('reason'),str) or not agreement['reason'].strip():
                raise ValueError('Invalid answer-equivalence verdict')
        if not agreement['equivalent']:
            raise ValueError('Independent answer mismatch: generated='+result['answer']+' source='+str(base['answer']['value']))
        independent_answer = result['answer']
        attempts.append(('normal',events,result['answer']))
    except (ValueError,KeyError,TypeError) as exc:
        yield {'source_id':identity,'variant':'normal','reject_reason':str(exc),
               'retry_feedback':{'previous_proposal':generation['response'],'check':'The proposal failed the required format or independent answer checks. Recheck the visible evidence and JSON schema.'}}
    if include_source and base.get('source_events'):
        if independent_answer is None:
            yield {'source_id':identity,'variant':'self_correction','reject_reason':'Source answer not confirmed by independent image answer'}
        else:
            attempts.append(('self_correction',copy.deepcopy(base['source_events']),independent_answer))
    for variant,events,answer in attempts:
        crops = [image.crop(e['box']) for e in events if e['type']=='visual_read']
        if variant=='self_correction':
            rewrite = teacher.generate([image]+crops,
                'Rewrite only the natural-language events in this visual reasoning trajectory. '
                'Preserve the original observation order, genuine initial mistaken belief or selection, its recognition, and the supported correction. '
                'Use concise natural first-person reasoning. Remove file paths, Python/code/tool instructions, Markdown headings, '
                'references to the user, and claims about externally provided coordinates. Do not invent new evidence or hide the correction. '
                'Return ONLY JSON {"texts":["first rewritten reasoning","second rewritten reasoning","third rewritten reasoning"]}. '
                'Return one string for EVERY reasoning_text event, in its original order, each in one or two concise sentences. '
                'Do not return visual events or event indices. '
                'Image 1 is original; remaining images are the actual observations. Question: '+base['query']+
                '\nOriginal events: '+json.dumps(events,ensure_ascii=False),max_tokens)
            rewrite.update(source_id=identity,stage='rewrite_source_correction',variant=variant)
            raw_log.write(json.dumps(rewrite,ensure_ascii=False)+'\n');raw_log.flush()
            try:
                if rewrite['finish_reason']!='eos':raise ValueError('Source rewrite truncated')
                events = rewrite_text_events(events,json_object(rewrite['response']))
            except (ValueError,KeyError,TypeError) as exc:
                yield {'source_id':identity,'variant':variant,'reject_reason':'Source rewrite rejected: '+str(exc)}
                continue
        crop_verdict = None
        if crops and crop_check:
            evidence = teacher.generate(crops,
                'These images are observation windows in reading order. You are NOT given the full original image. '
                'Describe what is actually visible in each crop, including exact readable text or object attributes. '
                'Solve the question only if these crops contain sufficient evidence; do not guess missing values. '
                'For a yes/no conjunction, one visibly false required property is sufficient to answer no; distinguish visible contrary evidence from a merely hidden property. '
                'Return ONLY JSON {"answerable":true,"answer":"short answer; option letter if requested",'
                '"observations":[{"window":1,"usable":true,"adds_new_evidence":true,"evidence":"visible contents and what this adds over earlier windows"}]}. '
                'For each window decide whether it supplies necessary visual information absent from previous windows. '
                'All crops come from the same original resolution: moving an overlapping window does not make an already complete target clearer. '
                'If it only repeats already visible objects or data, mark adds_new_evidence=false. '
                'Mark unusable crops (blank, clipped text, missing target) usable=false. '+
                f'There are exactly {len(crops)} images; return exactly {len(crops)} observation entries. '+
                ('Earlier windows may be incorrect selections; the final window must provide usable evidence. ' if variant=='self_correction' else '')+
                'Question: '+base['query'], max_tokens)
            evidence.update(source_id=identity,stage='crop_only',variant=variant)
            raw_log.write(json.dumps(evidence,ensure_ascii=False)+'\n');raw_log.flush()
            try:
                if evidence['finish_reason']!='eos':raise ValueError('Crop-only check truncated')
                crop_verdict = json_object(evidence['response'])
                validate_crop_evidence(crop_verdict, len(crops), answer, allow_early_errors=variant=='self_correction',require_novelty=True)
            except (ValueError, KeyError, TypeError) as exc:
                yield {'source_id':identity,'variant':variant,'reject_reason':'Crop-only evidence rejected: '+str(exc),
                    'retry_feedback':{'previous_proposal':generation['response'],'crop_only_check':evidence['response']}}
                continue
        quality = teacher.generate([image]+crops,check_prompt(base,answer,events,variant),max_tokens)
        quality.update(source_id=identity,stage='quality',variant=variant)
        raw_log.write(json.dumps(quality,ensure_ascii=False)+'\n');raw_log.flush()
        try:
            if quality['finish_reason']!='eos':raise ValueError('Quality check truncated')
            verdict = json_object(quality['response'])
            verify_verdict(verdict,base['answer'],answer,events,variant,semantic_agreement=agreement['equivalent'])
            candidate = {**base,'events':events,'variant':variant,'visual_only_safe':variant=='normal' and verdict['visual_only_safe'],
                'visual_only_basis':verdict['reason'],'annotation_origin':'teacher' if variant=='normal' else 'mixed',
                'metadata':{**base.get('metadata',{}),'curriculum':curriculum(events,variant),
                   'automatic_quality':{'passed':True,'verdict':verdict,'teacher':teacher.info,'independent_answer':answer,
                       'crop_only_verdict':crop_verdict,
                       'source_answer_agreement':agreement,
                       'evidence_type':'original_image_and_pixel_crops','student_feature_check':'not_performed'},
                   'generation':{'teacher':teacher.info,'source_answer_provided':variant=='self_correction',
                       'independent_answer_source_answer_provided':False,'variant':variant,
                       'trajectory_origin':'teacher' if variant=='normal' else 'source_teacher_rewrite', 'original_source_id':identity}}}
            candidate['metadata']['generation']['window_proposals']={'method':method,'boxes':windows}
            candidate['metadata']['generation']['retried'] = retry_feedback is not None
            if not events: candidate['direct_answer']=True
            records=build_records(candidate,dataset,run/'images',data_root)
            register(records,run.parent/'split_registry.sqlite')
            for record in records: yield record
        except (ValueError,KeyError,TypeError,OSError) as exc:
            yield {'source_id':identity,'variant':variant,'reject_reason':str(exc),
                'retry_feedback':{'previous_proposal':generation['response'],'quality_check':quality['response']}}


def rewrite_text_events(events: list[dict], response: dict) -> list[dict]:
    texts = response.get('texts')
    expected = [i for i,e in enumerate(events) if e['type']=='reasoning_text']
    if not isinstance(texts,list) or len(texts)!=len(expected):
        raise ValueError('Every source reasoning event must be rewritten exactly once')
    if all(isinstance(text,str) for text in texts):
        texts = [{'event_index':index,'text':text} for index,text in zip(expected,texts)]
    if [entry.get('event_index') for entry in texts]!=expected or any(type(entry.get('event_index')) is not int for entry in texts):
        raise ValueError('Source reasoning event indices changed')
    rewritten = copy.deepcopy(events)
    for entry in texts:
        if not isinstance(entry.get('text'),str) or not entry['text'].strip():
            raise ValueError('Empty rewritten source reasoning')
        rewritten[entry['event_index']]['text'] = entry['text']
    return rewritten


def validate_crop_evidence(verdict: dict, count: int, answer: str, allow_early_errors: bool = False, require_novelty: bool = False) -> None:
    if verdict.get('answerable') is not True or normalized_answer(verdict.get('answer','')) != normalized_answer(answer):
        raise ValueError('Crops alone did not independently support the same answer')
    observations = verdict.get('observations')
    if not isinstance(observations, list) or len(observations)!=count:
        raise ValueError('Every actual crop must be checked')
    for index, observation in enumerate(observations, 1):
        if type(observation.get('window')) is not int or observation['window']!=index or type(observation.get('usable')) is not bool:
            raise ValueError('Invalid observation check')
        if not observation['usable'] and (not allow_early_errors or index==count):
            raise ValueError('At least one observation window is unusable')
        if not isinstance(observation.get('evidence'),str) or not observation['evidence'].strip():
            raise ValueError('Crop evidence description is missing')
        if require_novelty and type(observation.get('adds_new_evidence')) is not bool:
            raise ValueError('Missing new-evidence check')
        if not allow_early_errors and (observation.get('adds_new_evidence') is False or any(
                normalized_answer(previous['evidence'])==normalized_answer(observation['evidence']) for previous in observations[:index-1])):
            raise ValueError('Observation repeats already available evidence')


def process_with_retry(base, dataset, teacher, run, data_root, max_tokens, raw_log, window_method, max_retries):
    feedback = None
    for attempt in range(max_retries+1):
        next_feedback = None
        normal_accepted = False
        for result in process_task(base,dataset,teacher,run,data_root,max_tokens,raw_log,
                include_source=attempt==0 and window_method!='direct',window_method=window_method,retry_feedback=feedback):
            if 'reject_reason' in result:
                result['attempt'] = attempt
                if result.get('variant')=='normal':
                    next_feedback = result.get('retry_feedback')
            elif result['metadata']['curriculum']['trajectory_type']=='normal':
                normal_accepted = True
            yield result
        if normal_accepted or next_feedback is None:
            break
        feedback = next_feedback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['cosyn','llava','thyme','visual_cot'],required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--model',default='Qwen3.5-9B')
    p.add_argument('--gpu',default='4')
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--scan-limit',type=int,default=120)
    p.add_argument('--per-shard-limit',type=int,default=5,help='Thyme source rows per completed shard; zero disables this cap')
    p.add_argument('--subsets',nargs='*',default=[])
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--max-pixels',type=int,default=1048576)
    p.add_argument('--window-method',choices=['catalog','coordinates','direct'],default='catalog')
    p.add_argument('--skip-images',type=int,default=0)
    p.add_argument('--max-retries',type=int,default=1,choices=[0,1],help='At most one normal-trajectory regeneration using check feedback')
    p.add_argument('--source-ids',nargs='*',default=[],help='Optional exact IDs for targeted small-sample checks')
    a=p.parse_args()
    if min(a.limit,a.scan_limit,a.max_new_tokens,a.max_pixels)<1:p.error('Budgets must be positive')
    if a.per_shard_limit<0:p.error('Per-shard limit must be nonnegative')
    if a.skip_images<0:p.error('Skipped-image count must be nonnegative')
    if a.output_root.exists():p.error('Use a fresh output directory')
    if a.output_root.resolve().is_relative_to(a.source_root.resolve()):p.error('Output must be outside source downloads')
    a.output_root.resolve().relative_to(a.data_root.resolve())
    tasks,failures=collect_tasks(a.dataset,a.source_root,a.scan_limit,a.limit,a.subsets,per_shard_limit=a.per_shard_limit,skip_images=a.skip_images,source_ids=a.source_ids)
    if a.source_ids and set(a.source_ids)-{t['source_id'] for t in tasks}:
        raise ValueError('Requested IDs were not found within the selected shards and scan budget: '+str(set(a.source_ids)-{t['source_id'] for t in tasks}))
    if not tasks:raise ValueError('No source tasks available')
    a.output_root.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    manifest={'config':config,'selected_source_ids':[t['source_id'] for t in tasks],'status':'running','counts':{}}
    implementation = [Path(__file__),Path(__file__).with_name('local_teacher.py'),Path(__file__).with_name('window_catalog.py')]
    manifest['implementation_sha256'] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation}
    manifest_path=a.output_root/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    counts=Counter();handles={}
    try:
        teacher=LocalTeacher(a.model,a.cache_root,a.gpu,a.max_pixels)
        manifest['teacher']=teacher.info
        with (a.output_root/'teacher_calls.jsonl').open('x') as raw, (a.output_root/'rejections.jsonl').open('x') as rejects, (a.output_root/'candidates.jsonl').open('x') as all_records:
            for failure in failures:rejects.write(json.dumps(failure,ensure_ascii=False)+'\n')
            for base in tasks:
                counts['source_tasks']+=1
                for result in process_with_retry(base,a.dataset,teacher,a.output_root,a.data_root,a.max_new_tokens,raw,a.window_method,a.max_retries):
                    if 'reject_reason' in result:
                        rejects.write(json.dumps(result,ensure_ascii=False)+'\n');rejects.flush();counts['rejected_variants']+=1
                        print(result['source_id'],result.get('variant'),result['reject_reason'],flush=True)
                        continue
                    group=result['metadata']['curriculum']
                    name=group['trajectory_type']+'_'+group['read_group']+'_'+result['reasoning_mode']
                    if name not in handles:handles[name]=(a.output_root/(name+'.jsonl')).open('x')
                    line=json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n'
                    handles[name].write(line);handles[name].flush();all_records.write(line);all_records.flush();counts[name]+=1;counts['records']+=1
                    print(base['source_id'],name,'accepted by automatic checks',flush=True)
                manifest['counts']=dict(counts);manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        manifest['status']='complete' if counts['records'] else 'no_accepted_records'
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for handle in handles.values():handle.close()
        manifest['counts']=dict(counts);manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if not counts['records']:raise RuntimeError('No automatically accepted records; inspect rejection log')

if __name__=='__main__':main()
