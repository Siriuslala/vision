"""Export HF Parquet image bytes and lossless JSON metadata, without downloads."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from utils.project_paths import load_paths

BENCHES = {"babyvision": "BabyVision", "babyvision-gen": "BabyVision-Gen"}


def snapshot(path):
    path = Path(path).expanduser()
    if not (path / "snapshots").is_dir():
        return path
    ref = path / "refs/main"
    if ref.exists():
        result = path / "snapshots" / ref.read_text().strip()
        if result.is_dir():
            return result
    candidates = sorted((path / "snapshots").iterdir())
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous snapshots in {path}; specify a snapshot directory")
    return candidates[0]


def export(bench, cache_root, output_root):
    import pyarrow.parquet as pq
    from PIL import Image

    name = BENCHES[bench]
    source = snapshot(Path(cache_root) / f"datasets--UnipatAI--{name}")
    shards = sorted(source.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet files under {source}")
    destination = Path(output_root).expanduser().resolve() / name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose a new --output-root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=destination.parent))
    records, seen, schemas = [], set(), {}
    try:
        for shard in shards:
            table = pq.ParquetFile(shard)
            schemas[str(shard.relative_to(source))] = str(table.schema_arrow)
            split = shard.stem.split("-000")[0]
            for batch in table.iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    case_id = row['taskId']
                    if (split, case_id) in seen:
                        raise ValueError(f"Duplicate taskId {case_id} in {split}")
                    seen.add((split, case_id))
                    row['split'] = split
                    row['source_shard'] = str(shard.relative_to(source))
                    row['image_metadata'] = {}
                    for field in ('image', 'answerImage'):
                        if field not in row:
                            continue
                        item = row[field]
                        raw = item.get('bytes')
                        if raw is None:
                            raw = (shard.parent / item['path']).read_bytes()
                        with Image.open(io.BytesIO(raw)) as img:
                            fmt, size, mode = img.format, img.size, img.mode
                            img.verify()
                        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt, "." + fmt.lower())
                        relative = Path(split) / field / f"{case_id}{suffix}"
                        target = staging / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(raw)
                        row[field] = relative.as_posix()
                        row['image_metadata'][field] = {
                            'original_path': item.get('path'), 'format': fmt,
                            'width': size[0], 'height': size[1], 'mode': mode,
                            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                        }
                    records.append(row)
        for split in sorted({r['split'] for r in records}):
            with (staging / f'{split}.jsonl').open('w', encoding='utf-8') as handle:
                for row in records:
                    if row['split'] == split:
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        info = {
            'benchmark': name, 'source_snapshot': str(source), 'num_rows': len(records),
            'splits': dict(Counter(r['split'] for r in records)),
            'types': dict(Counter(r['type'] for r in records)),
            'subtypes': dict(Counter(r['subtype'] for r in records)),
            'answer_types': dict(Counter(r.get('ansType', 'image_generation') for r in records)),
            'case_ids': [r['taskId'] for r in records], 'parquet_schemas': schemas,
            'image_fields': ['image', 'answerImage'] if bench.endswith('-gen') else ['image'],
            'note': 'Original split names retained. taskId is not row index. Images retain original bytes.',
        }
        (staging / 'dataset_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
        shutil.copy2(source / 'README.md', staging / 'SOURCE_README.md')
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f'{name}: {len(records)} rows exported to {destination}', flush=True)
    return info


def main():
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=str.lower, choices=[*BENCHES, 'all'], default='all')
    parser.add_argument('--cache-root', default=paths['HF_HUB_CACHE'])
    parser.add_argument('--output-root', default=paths['DATA_DIR'])
    args = parser.parse_args()
    for bench in BENCHES if args.bench == 'all' else [args.bench]:
        export(bench, args.cache_root, args.output_root)


if __name__ == '__main__':
    main()
