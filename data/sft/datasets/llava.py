"""Extract single-image tasks and import separately reviewed ROI supervision."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

from data.sft.core.annotations import apply_annotation, load_annotations
from data.sft.core.parquet import iter_rows

# Explicit source registrations; unseen mixtures need a source-specific audit.
TASKS = {
    'CLEVR-Math(MathV360K)': 'spatial_reasoning',
    'FigureQA(MathV360K)': 'chart_qa',
    'GEOS(MathV360K)': 'spatial_reasoning',
    'GeoQA+(MathV360K)': 'spatial_reasoning',
    'Geometry3K(MathV360K)': 'spatial_reasoning',
    'IconQA(MathV360K)': 'general_vqa',
    'MapQA(MathV360K)': 'general_vqa',
    'MathV360K_TQA': 'general_vqa',
    'MathV360K_VQA-AS': 'general_vqa',
    'MathV360K_VQA-RAD': 'general_vqa',
    'PMC-VQA(MathV360K)': 'general_vqa',
    'Super-CLEVR(MathV360K)': 'spatial_reasoning',
    'TabMWP(MathV360K)': 'table_qa',
    'UniGeo(MathV360K)': 'spatial_reasoning',
    'VisualWebInstruct(filtered)': 'general_vqa',
}


def extract_answer(response: str, query: str) -> tuple[dict, str]:
    """Require explicit separation for reasoning-bearing assistant responses."""
    response = response.strip()
    if not response:
        raise ValueError('empty_assistant_answer')
    tagged = re.fullmatch(r'<think>.*?</think>\s*<answer>(.*?)</answer>', response, re.S)
    simple = re.fullmatch(r'The answer is\s+([^\n]+)', response, re.S)
    if tagged:
        value, method = tagged[1].strip(), 'explicit_answer_tag'
    elif simple and len(simple[1]) <= 80 and not re.search(r'because|therefore|let.s|<think>|<answer>', simple[1], re.I):
        value, method = simple[1].strip(), 'source_answer_prefix'
    elif len(response) <= 80 and not re.search(r'[\n<>]|therefore|because|let.s|step\s*\d|answer\s+is', response, re.I):
        value, method = response, 'short_direct_source_answer'
    else:
        # Long descriptions are valid final responses only for explicit caption requests.
        if re.search(r'^(describe (what you see|the (image|picture|figure))|provide a .*description)', query, re.I) and not re.search(r'<think>|<answer>|```', response):
            return {'type':'text','value':response}, 'source_caption'
        raise ValueError('ambiguous_final_answer_requires_source_review')
    if not value:
        raise ValueError('empty_final_answer')
    if re.search(r'\bChoices\s*:', query):
        labels = re.findall(r'^\(([A-Z])\)', query, re.M)
        label = value.rstrip('.')
        if not labels or label not in labels:
            raise ValueError('answer_not_a_declared_choice')
        return {'type':'choice','value':label}, method
    return {'type':'text','value':value}, method


def convert_task(row: dict, shard: str, revision: str) -> dict:
    subset = row.get('data_source')
    if not isinstance(subset, str) or not subset:
        raise ValueError('missing_data_source')
    if re.search(r'babyvision|babyvision.gen', subset, re.I):
        raise ValueError('external_evaluation_source')
    image = row.get('image')
    if not isinstance(image, dict) or not isinstance(image.get('bytes'), bytes) or not image['bytes']:
        raise ValueError('missing_embedded_single_image')
    if subset not in TASKS:
        raise ValueError('unregistered_source_subset')
    split = Path(shard).name.split('-')[0]
    if split != 'train':
        raise ValueError('non_training_source_split')
    conversation = row.get('conversations')
    # Dropping earlier turns can change the question meaning and expose an invalid target.
    if not isinstance(conversation,list) or len(conversation) != 2:
        raise ValueError('multiturn_or_malformed_conversation_requires_context_conversion')
    human, assistant = conversation
    if human.get('from') != 'human' or assistant.get('from') != 'gpt':
        raise ValueError('invalid_conversation_roles')
    prompt, response = human.get('value'), assistant.get('value')
    if not isinstance(prompt,str) or not isinstance(response,str) or prompt.count('<image>') != 1:
        raise ValueError('invalid_single_image_prompt')
    query = prompt.replace('<image>','').strip()
    if not query or not isinstance(row.get('id'),str) or not row['id']:
        raise ValueError('empty_query_or_source_id')
    answer, method = extract_answer(response,query)
    task_type = 'captioning' if method == 'source_caption' else TASKS[subset]
    if subset == 'IconQA(MathV360K)' and re.search(r'\bhow many\b', query, re.I):
        raise ValueError('counting_requires_reviewed_point_or_bbox_instance_metadata')
    return {'source_id':f'{subset}/{row["id"]}/turn0', 'source_split':split,
            'source_revision':revision, 'task_type':task_type, 'query':query,
            'answer':answer, 'image':image['bytes'], 'annotation_origin':'derived',
            'metadata':{'source_subset':subset, 'source_shard':shard,
                        'source_conversations':conversation, 'source_answer_extraction':method,
                        'conversion_version':'llava-roi-v1', 'source_turn_index':0,
                        'source_content_sha256':hashlib.sha256(repr(row['conversations']).encode()).hexdigest(),
                        'trajectory_origin':'separate_local_teacher_or_human_annotation'}}


def iter_tasks(source_root: Path, limit: int) -> Iterator[dict]:
    """Yield source tasks or auditable rejections; limit counts inspected source rows."""
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    for index, (shard,row) in enumerate(iter_rows(source_root,limit)):
        metadata_path = source_root / '.cache/huggingface/download' / (shard + '.metadata')
        info = metadata_path.read_text().splitlines() if metadata_path.is_file() else []
        stat = (source_root/shard).stat()
        revision = info[0] if info else f'local:{shard}:{stat.st_size}:{stat.st_mtime_ns}'
        try:
            yield convert_task(row,shard,revision)
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            yield {'source_id':f'{shard}:{row.get("id",index)}', 'reject_reason':str(exc)}


def iter_candidates(source_root: Path, limit: int, annotations: Path | None = None) -> Iterator[dict]:
    registry = load_annotations(annotations)
    for base in iter_tasks(source_root,limit):
        yield base if 'reject_reason' in base else apply_annotation(base,registry)
