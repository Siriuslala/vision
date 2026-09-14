"""Bounded iteration over completed Parquet shards with provenance."""
from pathlib import Path
import pyarrow.parquet as pq


def iter_rows(root: Path, limit: int):
    count = 0
    files = sorted(p for p in root.rglob('*.parquet') if not any(s.startswith('.') for s in p.relative_to(root).parts))
    if not files:
        raise ValueError(f'No completed Parquet shards under {root}')
    for path in files:
        before = path.stat()
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1):
            if limit and count >= limit:
                return
            row = batch.to_pylist()[0]
            yield path.relative_to(root).as_posix(), row
            count += 1
        after = path.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise ValueError(f'Source shard changed during iteration: {path}')
