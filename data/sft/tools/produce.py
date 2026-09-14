"""Produce bounded SFT candidates without mutating source downloads."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import inspect
import json
from pathlib import Path
import sys

from data.sft.core.common import SOURCES, build_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(SOURCES), required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--image-output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=10, help='Source records to inspect; 0 means all completed files')
    parser.add_argument('--image-cache', type=Path)
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--split-registry',type=Path,help='Shared across all sources; defaults beside image output directories')
    parser.add_argument('--download-images', action='store_true')
    args = parser.parse_args()
    args.split_registry = args.split_registry or args.image_output.parent / 'split_registry.sqlite'
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    if not args.source_root.is_dir():
        parser.error('source-root must be an existing directory')
    args.image_output.resolve().relative_to(args.data_root.resolve())
    source = args.source_root.resolve()
    for dest in (args.output_root,args.image_output):
        if dest.resolve().is_relative_to(source):
            parser.error('Production paths must be outside source downloads')
    if args.output_root.exists():
        parser.error('output-root already exists; choose a fresh run directory')
    adapter = importlib.import_module('data.sft.datasets.'+args.dataset)
    signature = inspect.signature(adapter.iter_candidates)
    extras = {key:getattr(args,key) for key in ('image_cache','audit_dir','annotations','download_images','data_root') if key in signature.parameters and getattr(args,key) is not None}
    args.output_root.mkdir(parents=True)
    counts = Counter()
    seen = set()
    inputs = [{'path':str(p.relative_to(source)), 'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(source.rglob('*')) if p.is_file() and not any(part.startswith('.') for part in p.relative_to(source).parts)]
    manifest = {'dataset':SOURCES[args.dataset], 'source_root':str(source),'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, 'input_inventory':inputs, 'status':'running', 'quality':'pending', 'note':'Inventory is a snapshot, not a claim of download completeness. Source adapters skip temporary filenames and report unreadable completed files.'}
    manifest_path = args.output_root/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    failed = None
    try:
        with (args.output_root/'candidates.jsonl').open('x') as output, (args.output_root/'rejections.jsonl').open('x') as rejected:
            for candidate in adapter.iter_candidates(args.source_root,args.limit,**extras):
                counts['source_candidates'] += 1
                if 'reject_reason' in candidate:
                    counts['rejected'] += 1
                    rejected.write(json.dumps(candidate,ensure_ascii=False,allow_nan=False)+'\n')
                    continue
                try:
                    records = build_records(candidate,args.dataset,args.image_output,args.data_root)
                    from data.sft.core.splits import register
                    register(records,args.split_registry)
                    if any(record['sample_id'] in seen for record in records):
                        raise ValueError('Duplicate source sample ID')
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    counts['rejected'] += 1
                    rejected.write(json.dumps({'source_id':candidate.get('source_id'),'reject_reason':str(exc)},ensure_ascii=False)+'\n')
                    continue
                for record in records:
                    seen.add(record['sample_id'])
                    output.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    counts[record['reasoning_mode']] += 1
                output.flush()
                counts['converted_source_candidates'] += 1
    except Exception as exc:
        failed = exc
        manifest['error'] = f'{type(exc).__name__}: {exc}'
    manifest.update(status='failed' if failed else ('complete' if counts['converted_source_candidates'] else 'no_usable_candidates'), counts=dict(counts))
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'counts':dict(counts),'output':str(args.output_root)},ensure_ascii=False))
    if failed:
        raise failed
    if not counts['converted_source_candidates']:
        sys.exit(3)


if __name__ == '__main__':
    main()
