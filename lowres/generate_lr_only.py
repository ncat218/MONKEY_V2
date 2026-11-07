#!/usr/bin/env python3
"""Bicubic LR2/LR4/LR8 PNG generation from the supplied CS431 reference."""
import argparse
import csv
from pathlib import Path

import PIL
from PIL import Image

SCALES = (2, 4, 8)
MANIFEST_FIELDS = (
    'source', 'scale', 'clean_width', 'clean_height',
    'lr_width', 'lr_height', 'lr_path', 'pillow_version',
)


def require_pillow():
    if PIL.__version__ != '11.3.0':
        raise RuntimeError('This protocol requires Pillow==11.3.0')


def bicubic(image, size):
    return image.resize(size, Image.Resampling.BICUBIC)


def generate_variants(clean, relative, output):
    """Yield the original manifest rows and the matching in-memory LR images."""
    for scale in SCALES:
        size = (max(1, clean.width // scale), max(1, clean.height // scale))
        image = bicubic(clean, size)
        destination = output / f'lr{scale}x' / relative.parent / f'{relative.name}.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format='PNG')
        yield (relative.as_posix(), scale, *clean.size, *size,
               destination.relative_to(output).as_posix(), PIL.__version__), image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require_pillow()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if not source.is_dir():
        raise ValueError(f'Input directory not found: {source}')
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the input directory')
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    images = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        raise ValueError('No images found')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_FIELDS)
        for path in images:
            with Image.open(path) as image:
                clean = image.convert('RGB')
            for row, _ in generate_variants(clean, path.relative_to(source), output):
                writer.writerow(row)
    print(f'Completed: {len(images)} images x 3 scales -> {output}')


if __name__ == '__main__':
    main()
