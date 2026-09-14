"""Persistent cross-source split and image identity checks."""
from pathlib import Path
import sqlite3


def register(records: list[dict], registry: Path) -> None:
    registry.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(registry,timeout=30) as db:
        db.execute('CREATE TABLE IF NOT EXISTS splits (identity TEXT PRIMARY KEY, split TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        for record in records:
            split=record['provenance']['split']
            identities=['image:'+record['image']['sha256'],'group:'+record['provenance']['group_id']]
            for identity in identities:
                existing=db.execute('SELECT split FROM splits WHERE identity=?',(identity,)).fetchone()
                if existing and existing[0]!=split:
                    raise ValueError(f'Cross-source split conflict for {identity}: {existing[0]} vs {split}')
                db.execute('INSERT OR IGNORE INTO splits VALUES (?,?)',(identity,split))
