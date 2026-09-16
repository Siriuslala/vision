"""Shared image, geometry and record construction for schema 1.1."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

TASKS = set('recognition captioning general_vqa grounding detection counting ocr document_qa table_qa chart_qa spatial_reasoning curve_tracing maze difference_detection visual_comparison puzzle'.split())
SOURCES = {'visual_cot': 'deepcs233/Visual-CoT', 'thyme': 'Kwai-Keye/Thyme-SFT', 'pixmo': 'allenai/pixmo-count', 'cosyn': 'allenai/CoSyn-400K', 'llava': 'lmms-lab/LLaVA-OneVision-Data'}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def pixel_box(box: list[float], width: int, height: int) -> dict:
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ValueError('Box requires four finite pixel coordinates')
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f'Invalid pixel box {box} for {width}x{height}')
    scale = max(width, height)
    return {'center': [(x0+x1)/2/scale, (y0+y1)/2/scale], 'size': [(x1-x0)/scale, (y1-y0)/scale]}


def store_image(value: bytes | Path, directory: Path, data_root: Path) -> dict:
    raw = value if isinstance(value, bytes) else Path(value).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Source requires explicit EXIF geometry conversion')
        width, height = image.size
        suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif'}.get(image.format)
        if suffix is None or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Unsupported or multi-frame source image')
    path = directory / (sha + suffix)
    rel = relative_path(path, data_root)
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Existing content-addressed image is corrupt')
    else:
        with path.open('xb') as output:
            output.write(raw)
    return {'image_id': sha, 'path': rel, 'width': width, 'height': height, 'sha256': sha}


def validate_record(record: dict, data_root: Path | None = None) -> None:
    required = {'schema_version','record_type','sample_id','task_type','image','query','answer','reasoning_mode','coordinate_system','initial_region','trajectory','verification','provenance','quality'}
    if not required <= record.keys() or record.keys() - required - {'metadata'}:
        raise ValueError('Missing or unknown record fields')
    if record['schema_version'] != '1.1' or record['coordinate_system'] != 'image_long_side_normalized_xy' or record['record_type'] != 'sft':
        raise ValueError('Unsupported SFT schema/coordinate system')
    if record['task_type'] not in TASKS or not isinstance(record['query'], str) or not record['query'].strip():
        raise ValueError('Invalid task/query')
    if record['reasoning_mode'] not in ('visual_only','interleaved'):
        raise ValueError('Invalid reasoning mode')
    image = record['image']
    width, height = image['width'], image['height']
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError('Invalid image dimensions')
    path = Path(image['path'])
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Image path must be relative to DATA_DIR')
    if data_root is not None:
        raw = (data_root / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != image['sha256']:
            raise ValueError('Image hash mismatch')
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.size != (width, height):
                raise ValueError('Image dimensions mismatch')
    scale = max(width, height)
    bounds = [width/scale, height/scale]
    expected = {'region_id':0, 'center':[v/2 for v in bounds], 'size':bounds}
    if record['initial_region'] != expected:
        raise ValueError('Invalid initial region')
    events = record['trajectory']
    if not events or events[-1] != {'event_id':len(events),'type':'finish','supervise':True}:
        raise ValueError('Missing terminal finish')
    regions = 0
    for index, event in enumerate(events, 1):
        if event['event_id'] != index or type(event['supervise']) is not bool:
            raise ValueError('Invalid event numbering/supervision')
        kind = event['type']
        if kind == 'visual_read':
            regions += 1
            if event['region_id'] != regions or type(event['reference_region_id']) is not int or not 0 <= event['reference_region_id'] < regions:
                raise ValueError('Invalid region reference')
            region = event['region']
            for center, size, bound in zip(region['center'],region['size'],bounds, strict=True):
                if not math.isfinite(center) or not math.isfinite(size) or size <= 0 or center-size/2 < -1e-6 or center+size/2 > bound+1e-6:
                    raise ValueError('Out-of-bounds window')
        elif kind == 'reasoning_text':
            if record['reasoning_mode'] == 'visual_only' or not event['text'].strip():
                raise ValueError('Invalid reasoning text')
        elif kind != 'finish' or index != len(events):
            raise ValueError('Invalid event type/order')
    answer = record['answer']
    if set(answer) != {'type','value'}:
        raise ValueError('Invalid answer fields')
    kind, value = answer['type'], answer['value']
    if kind in ('text','choice'):
        if not isinstance(value,str) or (kind == 'choice' and not value):
            raise ValueError('Invalid text/choice answer')
    elif kind == 'number':
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Invalid numeric answer')
    elif kind in ('points','polyline'):
        if not isinstance(value,list) or (kind == 'polyline' and len(value)<2):
            raise ValueError('Invalid geometric answer')
        for point in value:
            if len(point)!=2 or any(not math.isfinite(v) or not 0<=v<=b for v,b in zip(point,bounds)):
                raise ValueError('Invalid answer point')
    elif kind == 'boxes':
        for box in value:
            x0,y0,x1,y1 = box['xyxy']
            if not 0<=x0<x1<=bounds[0] or not 0<=y0<y1<=bounds[1]:
                raise ValueError('Invalid answer box')
    else:
        raise ValueError('Unknown answer type')
    if record['quality']['status'] not in ('pending','validated','example'):
        raise ValueError('Invalid quality status')
    json.dumps(record, allow_nan=False)
    validate_extensions(record)


def build_records(candidate: dict, dataset: str, image_output: Path, data_root: Path) -> list[dict]:
    if type(candidate.get('visual_only_safe',False)) is not bool:
        raise ValueError('visual_only_safe must be a boolean')
    image = store_image(candidate['image'], image_output, data_root)
    width, height = image['width'], image['height']
    scale = max(width,height)
    modes = ['interleaved'] + (['visual_only'] if candidate.get('visual_only_safe',False) else [])
    records = []
    for mode in modes:
        events, mapping, region_id = [], {}, 0
        for old_id, raw in enumerate(candidate['events'],1):
            if raw['type'] == 'reasoning_text' and mode == 'visual_only':
                continue
            event = {'event_id':len(events)+1,'type':raw['type'],'supervise':raw.get('supervise',True)}
            mapping[old_id] = event['event_id']
            if raw['type'] == 'visual_read':
                region_id += 1
                event.update(region_id=region_id, reference_region_id=raw.get('reference_region_id',0), region=pixel_box(raw['box'],width,height))
            elif raw['type'] == 'reasoning_text':
                event['text'] = raw['text']
            else:
                raise ValueError('Adapter events may contain only reads and reasoning text')
            events.append(event)
        if not region_id and not candidate.get('direct_answer',False):
            raise ValueError('No ROI evidence without an explicit direct-answer decision')
        if not region_id and events:
            raise ValueError('Direct-answer examples must not contain intermediate reasoning')
        events.append({'event_id':len(events)+1,'type':'finish','supervise':True})
        metadata = copy.deepcopy(candidate.get('metadata',{}))
        for derivation in metadata.get('window_derivations',[]):
            derivation['event_id'] = mapping[derivation['event_id']]
        if 'counting' in metadata:
            counting = metadata['counting']
            counting['sum_event_id'] = mapping.get(counting['sum_event_id'])
            for group in counting['groups']:
                group['read_event_id'] = mapping[group['read_event_id']]
                group['count_event_id'] = mapping.get(group['count_event_id'])
        metadata['production'] = {'adapter':dataset, 'version':'1', 'visual_only_basis':candidate.get('visual_only_basis','adapter-reviewed operation dependency') if mode=='visual_only' else None}
        source_id = str(candidate['source_id'])
        group = candidate.get('group_id',image['sha256'])
        source_split = candidate.get('source_split','train')
        split = 'validation' if source_split in ('val','validation','dev') else 'test' if source_split=='test' else ('validation' if int(digest(group)[:8],16)%100 < 5 else 'train')
        record = {'schema_version':'1.1','record_type':'sft','sample_id':dataset+':'+digest([source_id,mode]+([candidate['variant']] if 'variant' in candidate else []))[:24], 'task_type':candidate['task_type'], 'image':image,'query':candidate['query'], 'answer':copy.deepcopy(candidate['answer']), 'reasoning_mode':mode,'coordinate_system':'image_long_side_normalized_xy','initial_region':{'region_id':0,'center':[width/scale/2,height/scale/2],'size':[width/scale,height/scale]},'trajectory':events,'verification':candidate.get('verification'), 'provenance':{'source_name':SOURCES[dataset],'source_revision':candidate.get('source_revision','local-unversioned'),'source_sample_id':source_id+('/variant/'+candidate['variant'] if 'variant' in candidate else ''),'source_split':source_split,'split':split,'group_id':str(group),'annotation_origin':candidate.get('annotation_origin','derived'),'conversion_version':'1','seed':0},'quality':{'status':'pending','notes':'Automatically checked where metadata records the result; awaiting final sample inspection and batch acceptance.'},'metadata':metadata}
        validate_record(record)
        records.append(record)
    return records


def _fields(value: dict, required: set, optional: set = frozenset()) -> None:
    if not isinstance(value,dict) or not required <= value.keys() or value.keys()-required-optional:
        raise ValueError(f'Invalid object fields; expected {sorted(required)}')


def _nonempty(value: Any) -> bool:
    return isinstance(value,str) and bool(value.strip())


def _path(value: Any) -> bool:
    return _nonempty(value) and not Path(value).is_absolute() and '..' not in Path(value).parts and '\\' not in value


def validate_extensions(record: dict) -> None:
    """Validate nested schema and counting supervision before admission."""
    import re
    image=record['image']
    _fields(image,{'image_id','path','width','height','sha256'})
    if not _nonempty(image['image_id']) or not _path(image['path']) or not re.fullmatch('[0-9a-f]{64}',str(image['sha256'])):
        raise ValueError('Invalid image identity/path/hash')
    if not _nonempty(record['sample_id']):raise ValueError('Empty sample ID')
    _fields(record['quality'],{'status','notes'})
    if not isinstance(record['quality']['notes'],str):raise ValueError('Invalid quality notes')
    provenance=record['provenance']
    _fields(provenance,{'source_name','source_revision','source_sample_id','source_split','split','group_id','annotation_origin','conversion_version','seed'})
    if any(not _nonempty(provenance[k]) for k in ('source_name','source_revision','source_sample_id','source_split','group_id','conversion_version')):
        raise ValueError('Invalid provenance identity')
    if provenance['split'] not in ('train','validation','test') or provenance['annotation_origin'] not in ('human','program','teacher','derived','mixed'):
        raise ValueError('Invalid provenance split/origin')
    if provenance['seed'] is not None and (type(provenance['seed']) is not int or provenance['seed']<0):raise ValueError('Invalid seed')
    verification=record['verification']
    if verification is not None:
        _fields(verification,{'verifier_id','version','params'})
        if not _nonempty(verification['verifier_id']) or not _nonempty(verification['version']) or not isinstance(verification['params'],dict):raise ValueError('Invalid verification')
    events=record['trajectory']
    for event in events:
        kind=event['type']
        required={'event_id','type','supervise'} | ({'region_id','reference_region_id','region'} if kind=='visual_read' else {'text'} if kind=='reasoning_text' else set())
        _fields(event,required,{'annotation'})
        if type(event['event_id']) is not int or ('annotation' in event and not isinstance(event['annotation'],str)):raise ValueError('Invalid event types')
        if kind=='visual_read':
            _fields(event['region'],{'center','size'})
            if type(event['region_id']) is not int or any(not isinstance(event['region'][k],list) or len(event['region'][k])!=2 for k in ('center','size')):raise ValueError('Invalid region shape')
            if any(type(v) not in (int,float) for k in ('center','size') for v in event['region'][k]):raise ValueError('Invalid geometry number')
    if record['answer']['type']=='boxes':
        if not isinstance(record['answer']['value'],list):raise ValueError('Boxes require a list')
        for box in record['answer']['value']:
            _fields(box,{'label','xyxy'})
            if not isinstance(box['label'],str):raise ValueError('Invalid box label')
    metadata=record.get('metadata',{})
    if not isinstance(metadata,dict):raise ValueError('Metadata requires object')
    def event_at(index, kind):
        if type(index) is not int or not 1<=index<=len(events) or events[index-1]['type']!=kind:
            raise ValueError('Metadata event reference mismatch')
        return events[index-1]
    derivations=metadata.get('window_derivations',[])
    if not isinstance(derivations,list):raise ValueError('Invalid derivation list')
    seen=set()
    for d in derivations:
        _fields(d,{'event_id','method','config_id','source_point_ids','audit_record_path'})
        event_at(d['event_id'],'visual_read')
        if d['event_id'] in seen or d['method'] not in ('geometry','sam3','geometry_sam3') or not _nonempty(d['config_id']) or not _path(d['audit_record_path']):raise ValueError('Invalid derivation metadata')
        seen.add(d['event_id'])
        if not isinstance(d['source_point_ids'],list) or not d['source_point_ids'] or not all(_nonempty(v) for v in d['source_point_ids']):raise ValueError('Invalid source points')
    if record['task_type']!='counting' and 'counting' not in metadata:return
    # Counting tasks with semantic additions are not instance-group supervision.
    if 'counting' not in metadata:
        if record['task_type']=='counting':raise ValueError('Counting process requires verified instance-group metadata')
        return
    c=metadata['counting']
    _fields(c,{'config_id','mode','instance_order','audit_record_path','groups','sum_event_id'})
    if c['mode'] not in ('exclusive','new_instances') or not _nonempty(c['config_id']) or not _path(c['audit_record_path']):raise ValueError('Invalid counting configuration')
    def unique_ids(values):
        if not isinstance(values,list) or any(not _nonempty(v) for v in values) or len(values)!=len(set(values)):raise ValueError('Invalid instance IDs')
        return set(values)
    expected=unique_ids(c['instance_order']);counted=set();last_read=0
    if not isinstance(c['groups'],list):raise ValueError('Invalid counting groups')
    for group in c['groups']:
        _fields(group,{'read_event_id','count_event_id','visible_instance_ids','new_instance_ids','local_count','new_count','cumulative_count'})
        event_at(group['read_event_id'],'visual_read')
        if group['read_event_id']<=last_read:raise ValueError('Counting group order mismatch')
        last_read=group['read_event_id']
        visible=unique_ids(group['visible_instance_ids']);new=unique_ids(group['new_instance_ids'])
        if not visible<=expected or new!=visible-counted or (c['mode']=='exclusive' and visible!=new):raise ValueError('Counting overlap/coverage mismatch')
        counted|=new
        for field,value in [('local_count',len(visible)),('new_count',len(new)),('cumulative_count',len(counted))]:
            if type(group[field]) is not int or group[field]!=value:raise ValueError('Incorrect local/cumulative count')
        if record['reasoning_mode']=='visual_only':
            if group['count_event_id'] is not None:raise ValueError('Pure visual count text reference')
        else:
            event_at(group['count_event_id'],'reasoning_text')
            if group['count_event_id']!=group['read_event_id']+1:raise ValueError('Counting text must follow observation')
    if counted!=expected or record['answer']!={'type':'number','value':len(counted)}:raise ValueError('Final count/instance coverage mismatch')
    if record['reasoning_mode']=='visual_only':
        if c['sum_event_id'] is not None:raise ValueError('Pure visual sum text reference')
    elif c['groups']:
        event_at(c['sum_event_id'],'reasoning_text')
        if c['sum_event_id']<=last_read:raise ValueError('Sum precedes observations')
