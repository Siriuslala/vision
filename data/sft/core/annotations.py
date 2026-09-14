"""Human/teacher evidence annotations keyed to immutable source content."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data.sft.core.common import digest


def source_key(source_id: str, query: str, answer: dict, image: bytes) -> str:
    return digest([source_id, query, answer, hashlib.sha256(image).hexdigest()])


def load_annotations(path: Path | None) -> dict:
    if path is None:
        return {}
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = row['source_key']
        if key in result:
            raise ValueError(f'Duplicate annotation key {key}')
        result[key] = row
    return result


def apply_annotation(base: dict, annotations: dict) -> dict:
    key = source_key(base['source_id'],base['query'],base['answer'],base['image'])
    annotation = annotations.get(key)
    if annotation is None:
        return {'source_id':base['source_id'],'reject_reason':'missing_region_annotation','source_key':key}
    if annotation.get('status') == 'rejected':
        if not annotation.get('reason'):raise ValueError('Rejected annotation requires reason')
        return {'source_id':base['source_id'],'source_key':key,'reject_reason':annotation['reason']}
    if annotation.get('status') not in ('proposed','reviewed'):
        raise ValueError('Annotation status must be proposed or reviewed')
    if not annotation.get('events'):
        raise ValueError('Annotation requires nonempty events')
    if annotation.get('status') == 'reviewed' and not annotation.get('reviewer'):
        raise ValueError('Reviewed annotations require reviewer identity')
    safe=annotation.get('visual_only_safe',False)
    if type(safe) is not bool or (safe and (annotation['status']!='reviewed' or not annotation.get('visual_only_basis'))):
        raise ValueError('visual_only requires a boolean and reviewed dependency rationale')
    return {**base,'events':annotation['events'], 'visual_only_safe':safe, 'visual_only_basis':annotation.get('visual_only_basis',''), 'annotation_origin':'mixed','metadata':{**base.get('metadata',{}), 'region_annotation':{k:v for k,v in annotation.items() if k!='events'}}}
