"""Checked, fsynced writes for new recovery artifacts; frozen helpers stay unchanged."""
import os
from pathlib import Path
import tempfile

from lowres.common import canonical


def write_bytes(path,payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=bytes(payload)
    if not payload:raise ValueError('Refuse an empty recovery artifact')
    descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            written=stream.write(payload)
            if written!=len(payload):raise OSError('Short artifact write')
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes()!=payload:raise OSError('Artifact read-back differs before rename')
        os.replace(temporary,path)
        if path.read_bytes()!=payload:raise OSError('Artifact read-back differs after rename')
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path,value):
    write_bytes(path,canonical(value))


def write_text(path,value):
    write_bytes(path,value.encode('utf-8'))


def storage_probe(root):
    root=Path(root)
    target=root/'storage_probe_payload.json'
    payload=dict(probe='checked fsync and read-back',data='0123456789abcdef'*65536)
    save_json(target,payload)
    target.unlink()
    save_json(root/'storage_probe.json',dict(status='PASS',payload_bytes=len(canonical(payload)),
                                           flush=True,fsync=True,readback=True))
