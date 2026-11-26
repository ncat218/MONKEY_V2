"""Mix A / Mix B tools for the CS431 MonkeyOCRv2 native-LR2 study.

This package lives in the member's own workspace/repo. Tuấn's MONKEY_V2 repository is only a
pipeline reference: CPU tools here are self-contained, and anything that must reuse MONKEY_V2 code
(inference, evaluator, quad grid) imports it through `mixab.monkey_janek` with its commit recorded.
"""
import hashlib
import json
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    """Same canonical JSON bytes as MONKEY_V2 lowres.common.canonical."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(canonical(value))
    os.replace(temp, path)
