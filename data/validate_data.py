"""Check every exported image against its source Parquet bytes and recorded hash."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from data.prepare_data import BENCHES
from utils.project_paths import load_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default=load_paths()['DATA_DIR'])
    p.add_argument('--bench', choices=[*BENCHES, 'all'], default='babyvision')
    args = p.parse_args()
    names = BENCHES.values() if args.bench == 'all' else [BENCHES[args.bench]]
    for name in names:
        root = Path(args.data_root) / name
        info = json.loads((root / 'dataset_info.json').read_text())
        source = Path(info['source_snapshot'])
        original = {}
        for shard in source.rglob('*.parquet'):
            for row in pq.read_table(shard).to_pylist():
                original[row['taskId']] = row
        count = image_count = 0
        ids = set()
        for split, expected in info['splits'].items():
            rows = [json.loads(line) for line in (root / f'{split}.jsonl').read_text().splitlines()]
            assert len(rows) == expected
            for row in rows:
                assert row['taskId'] not in ids
                ids.add(row['taskId'])
                count += 1
                for key, value in original[row['taskId']].items():
                    if key not in info['image_fields']:
                        assert row[key] == value, (row['taskId'], key)
                for field in info['image_fields']:
                    raw = (root / row[field]).read_bytes()
                    assert raw == original[row['taskId']][field]['bytes']
                    assert hashlib.sha256(raw).hexdigest() == row['image_metadata'][field]['sha256']
                    with Image.open(root / row[field]) as im:
                        im.load()
                        assert im.size == (row['image_metadata'][field]['width'], row['image_metadata'][field]['height'])
                    image_count += 1
        assert count == info['num_rows'] == len(original)
        print(f'{name}: {count} records and {image_count} images verified against original Parquet')


if __name__ == '__main__':
    main()
