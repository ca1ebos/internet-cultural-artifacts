import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'
CONFIG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
SOURCES = json.loads((ROOT / 'sources.json').read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.part')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(pending, path)


def read_rows(path):
    with path.open(encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.part')
    with pending.open('w', encoding='utf-8', newline='\n') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(pending, path)
