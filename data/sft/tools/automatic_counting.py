"""Reannotate changed PixMo images explicitly, with independent teacher checks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from data.sft.core.common import build_records
from data.sft.core.splits import register
from data.sft.datasets.pixmo import counting_candidate, rows
from data.sft.tools.automatic import check_prompt, curriculum, verify_verdict
from data.sft.tools.local_teacher import LocalTeacher, json_object
from data.sft.tools.point_windows import file_sha256, select_mask_candidate


def approved_instances(proposal: dict) -> list[dict]:
    instances = []
    for index, candidate in enumerate(proposal['candidate_instances']):
        if candidate['point_index'] != index:
            raise ValueError('SAM results must preserve source point order')
        # SAM scores rank proposals; semantic acceptance comes from the teacher.
        # Low scores are retained for inspection, never promoted as audited boxes.
        selected = select_mask_candidate(candidate['rounds'], minimum_score=0.0)
        if selected['status'] != 'requires_semantic_verification':
            raise ValueError(f'Point {index} has no stable SAM candidate: {selected["reason"]}')
        instances.append({'instance_id': f'point_{index}', 'box': selected['box']})
    if len(instances) != proposal['count']:
        raise ValueError('SAM instance count disagrees with proposed count')
    return instances


def check_independent_count(result: dict, instance_count: int) -> None:
    if result.get('answerable') is not True or type(result.get('count')) is not int:
        raise ValueError('Teacher cannot confidently count this image')
    if result['count'] != instance_count:
        raise ValueError(f'Independent count {result["count"]} disagrees with {instance_count} instances')
    if not isinstance(result.get('evidence'), str) or not result['evidence'].strip():
        raise ValueError('Independent count lacks visual evidence')


def check_instances(result: dict, count: int) -> None:
    if any(result.get(key) is not True for key in ('all_instances_correct', 'complete', 'one_instance_per_box')):
        raise ValueError('Teacher rejected instance semantics, boundaries or completeness')
    decisions = result.get('instances')
    if not isinstance(decisions, list) or len(decisions) != count:
        raise ValueError('Teacher must verify every instance')
    if {x.get('id') for x in decisions} != {f'point_{i}' for i in range(count)}:
        raise ValueError('Teacher instance IDs do not match proposed boxes')
    for decision in decisions:
        if any(decision.get(key) is not True for key in ('label_correct', 'one_instance', 'visible_extent_complete')):
            raise ValueError('Individual instance failed teacher checks')
        if not isinstance(decision.get('evidence'), str) or not decision['evidence'].strip():
            raise ValueError('Instance verdict lacks evidence')


def process(proposal: dict, row: dict, revision: str, teacher, run: Path, data_root: Path, log, max_tokens: int):
    source_id = proposal['source_id']
    image_path = Path(proposal['image_path'])
    actual = file_sha256(image_path)
    if actual != proposal['image_sha256'] or row['image_sha256'] != proposal['source_image_sha256']:
        raise ValueError('Diagnostic image or source identity changed')
    proposal_points = [p['point'] if isinstance(p, dict) else p for p in proposal['points']]
    if list(map(list, zip(row['points']['x'], row['points']['y']))) != proposal_points:
        raise ValueError('Diagnostic points do not match original source')
    if row['count'] != proposal['count'] or row['label'] != proposal['label']:
        raise ValueError('Diagnostic count or label changed')
    image = Image.open(image_path).convert('RGB')
    def ask(images, prompt, stage):
        raw = teacher.generate(images, prompt, max_tokens)
        raw.update(source_id=source_id, stage=stage)
        log.write(json.dumps(raw, ensure_ascii=False)+'\n'); log.flush()
        if raw['finish_reason'] != 'eos':
            raise ValueError('Teacher output truncated')
        return json_object(raw['response'])
    # No source count, source points or SAM outputs are given to this call.
    independent = ask([image], 'Count all visible '+row['label']+' in this image independently. '
        'Do not infer hidden objects. If uncertain set answerable=false. Return only JSON '
        '{"answerable":true,"count":0,"evidence":"describe the visible targets and their locations"}.', 'independent_count')
    check_independent_count(independent, proposal['count'])
    instances = approved_instances(proposal)
    overlay = image.copy(); draw = ImageDraw.Draw(overlay)
    for instance in instances:
        box = instance['box']; draw.rectangle(box, outline='red', width=3)
        draw.text((box[0], box[1]), instance['instance_id'], fill='red', stroke_width=1, stroke_fill='white')
    crops = [image.crop(instance['box']) for instance in instances]
    crop_checks = []
    for instance, crop in zip(instances, crops):
        result = ask([crop], 'This is one cropped candidate instance, without the original image. '
            'Check whether it shows exactly one '+row['label']+' with the complete visible body or object extent. '
            'Reject a body part, unrelated object, crop-cut extremities or multiple target instances. '
            'Return only JSON {"label_correct":true,"one_instance":true,"visible_extent_complete":true,"evidence":"describe what is actually visible"}.',
            'crop_only_'+instance['instance_id'])
        if any(result.get(key) is not True for key in ('label_correct','one_instance','visible_extent_complete')):
            raise ValueError('Crop-only instance rejection: '+instance['instance_id']+' '+str(result))
        if not isinstance(result.get('evidence'),str) or not result['evidence'].strip():
            raise ValueError('Crop-only instance evidence missing')
        crop_checks.append({'id':instance['instance_id'],**result})
    semantic = ask([image, overlay]+crops, 'Verify a counting annotation for '+row['label']+'. '
        'Image 1 is the untouched original, image 2 shows numbered candidate boxes, and remaining images '
        'are those crops in point_0, point_1, ... order. Check all visible target instances are covered exactly once. '
        'Each crop must show one target with its complete visible extent, not a body part. '
        'Return only JSON with boolean all_instances_correct, complete, one_instance_per_box, and instances: '
        '[{"id":"point_0","label_correct":true,"one_instance":true,"visible_extent_complete":true,"evidence":"specific visual evidence"}]. '
        'For zero targets return instances=[].', 'instance_verification')
    check_instances(semantic, len(instances))
    config = {'version':'pixmo-independent-reannotation-v1', 'teacher':teacher.info,
              'source_revision':revision, 'source_id':source_id, 'original_image_sha256':row['image_sha256'],
              'actual_image_sha256':actual, 'instances':instances, 'proposal':proposal}
    config_id = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    audit = {'source_id':source_id,'image_sha256':actual,'method':'sam3','config_id':config_id,
             'instance_boxes_reviewed':True,'complete_instance_set_reviewed':True,'instances':instances,
             'independent_count':independent,'semantic_verdict':semantic,'crop_only_checks':crop_checks,'teacher':teacher.info,
             'source_image_sha256':row['image_sha256'],'source_count':row['count']}
    audit_dir = run/'audits'; audit_dir.mkdir(exist_ok=True)
    audit_path = audit_dir/(source_id.replace(':','_')+'.json')
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (audit_dir/(source_id.replace(':','_')+'_config.json')).write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    revised_row = copy.deepcopy(row); revised_row['image_sha256'] = actual
    candidate = counting_candidate(revised_row,source_id,revision+':reannotation:'+config_id,image_path,
                                   audit,audit_path.relative_to(data_root).as_posix())
    # Re-encoded versions of the same source image must share the split group.
    candidate['group_id'] = row['image_sha256']
    verdict = ask([image]+[image.crop(e['box']) for e in candidate['events'] if e['type']=='visual_read'],
                  check_prompt(candidate,str(row['count']),candidate['events'],'normal'),'trajectory_verification')
    verify_verdict(verdict,candidate['answer'],str(independent['count']),candidate['events'],'normal')
    candidate.update(annotation_origin='mixed',variant='normal',visual_only_safe=verdict['visual_only_safe'],
                     visual_only_basis=verdict['reason'])
    candidate['metadata'].update(curriculum=curriculum(candidate['events'],'normal'),
        automatic_quality={'passed':True,'verdict':verdict,'teacher':teacher.info,'student_feature_check':'not_performed'},
        image_reannotation={'original_sha256':row['image_sha256'],'actual_sha256':actual,
            'source_revision':revision,'source_count':row['count'],'source_url':row['image_url'],
            'independent_count':independent,'semantic_verdict':semantic,'crop_only_checks':crop_checks,'teacher':teacher.info,
            'identity':'explicit_new_image_revision','source_byte_hash_matched':actual==row['image_sha256']})
    records = build_records(candidate,'pixmo',run/'images',data_root)
    register(records,run.parent/'split_registry.sqlite')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('proposal-root','source-root','output-root','data-root','cache-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',default='Qwen3.5-9B');p.add_argument('--gpu',default='4')
    p.add_argument('--limit',type=int,default=5);p.add_argument('--max-new-tokens',type=int,default=2048)
    a=p.parse_args()
    if a.limit<1:p.error('Use a positive small-sample limit')
    if a.output_root.exists():p.error('Use a fresh output directory')
    a.output_root.resolve().relative_to(a.data_root.resolve())
    proposals=[json.loads(path.read_text()) for path in sorted(a.proposal_root.glob('*.json')) if path.name != 'config.json'][:a.limit]
    wanted={proposal['source_id'] for proposal in proposals}
    source={}
    for source_id,row,revision in rows(a.source_root,0):
        if source_id in wanted:source[source_id]=(row,revision)
        if wanted.issubset(source):break
    if wanted-set(source):raise ValueError('Missing source records')
    a.output_root.mkdir(parents=True)
    teacher=LocalTeacher(a.model,a.cache_root,a.gpu)
    accepted=0;rejected=0
    with (a.output_root/'teacher_calls.jsonl').open('x') as log, (a.output_root/'candidates.jsonl').open('x') as output, (a.output_root/'rejections.jsonl').open('x') as rejects:
        for proposal in proposals:
            try:
                result=process(proposal,*source[proposal['source_id']],teacher,a.output_root,a.data_root,log,a.max_new_tokens)
                for record in result:
                    line=json.dumps(record,ensure_ascii=False)+'\n';output.write(line);output.flush()
                    group=record['metadata']['curriculum']
                    with (a.output_root/('normal_'+group['read_group']+'_'+record['reasoning_mode']+'.jsonl')).open('a') as stream:stream.write(line)
                    accepted+=1
                print(proposal['source_id'],'accepted',len(result),flush=True)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                rejects.write(json.dumps({'source_id':proposal['source_id'],'reason':str(exc)})+'\n');rejects.flush();rejected+=1
                print(proposal['source_id'],'rejected',str(exc),flush=True)
    (a.output_root/'manifest.json').write_text(json.dumps({'teacher':teacher.info,'records':accepted,'rejected_tasks':rejected,'status':'complete' if accepted else 'no_accepted_records'},indent=2)+'\n')
    if not accepted:raise RuntimeError('No counting records accepted; see rejection evidence')


if __name__=='__main__':main()
