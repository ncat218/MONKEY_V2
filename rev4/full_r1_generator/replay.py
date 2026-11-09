"""Replay the original generator in its own pinned image environment."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import io
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from lowres.common import jsonl, sha256 as file_sha256, write_jsonl
from rev4 import require
from . import CONFIG_SHA256, QA_SHA256, REQUIREMENTS, operators


def source_seed(source):
    # stable_seed from the historical generator serializes each field with '|'.
    parts = ('42', QA_SHA256, CONFIG_SHA256, 'train', source, 'realistic_mix', 'L2')
    digest = sha256('|'.join(parts).encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def geometry_params(size, seed):
    """Only perspective changes the common FOV. Recover its RNG state on a blank canvas."""
    rng = np.random.default_rng(seed)
    ops = operators.choose_mix('L2', rng)
    params = [dict(operation=op) for op in ops]
    if 'perspective' in ops:
        width, height = size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for i, op in enumerate(ops):
            canvas, values = operators.apply_op(canvas, op, 'L2', None, rng)
            if op == 'perspective':
                params[i].update(values)
                break
    return params


def fingerprint(row):
    return dict(source_image_id=row['source_image_id'], clean_sha256=file_sha256(row['path']),
                params=geometry_params(row['size'], source_seed(row['source_image_id'])))


def encode(row):
    source = row['source_image_id']
    data = Path(row['path']).read_bytes()
    require(sha256(data).hexdigest() == row['clean_sha256'], f'Clean bytes differ for {source}')
    with Image.open(io.BytesIO(data)) as image:
        require(list(image.size) == row['size'] and image.getexif().get(274, 1) in (None, 1),
                f'Clean dimensions or EXIF differ for {source}')
        clean = np.asarray(image.convert('RGB'), dtype=np.uint8)
    output, params = operators.degrade(clean, 'realistic_mix', 'L2', None, source_seed(source))
    ok, png = cv2.imencode('.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, 6])
    require(ok, f'Original PNG encoding failed for {source}')
    return dict(source_image_id=source, degraded_sha256=sha256(png).hexdigest(), params=params)


def single_thread():
    cv2.setNumThreads(1)


def main():
    mode, source, destination = sys.argv[1:]
    for requirement in REQUIREMENTS:
        name, expected = requirement.split('==')
        require(version(name) == expected, f'Historical generator requires {requirement}')
    task = {'geometry': fingerprint, 'png': encode}[mode]
    with ProcessPoolExecutor(max_workers=int(os.environ['SLURM_CPUS_PER_TASK']),
                             initializer=single_thread) as pool:
        def records():
            for index, row in enumerate(pool.map(task, jsonl(source), chunksize=4), 1):
                if index == 1 or index % 250 == 0:
                    print(f'FULL_R1 {mode} processed {index} sources', flush=True)
                yield row
        write_jsonl(destination, records())


if __name__ == '__main__':
    main()
