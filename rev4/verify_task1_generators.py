"""Verify frozen generator bytes and dry-run on two S1 images only."""
import argparse
import csv
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys

import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from rev4 import require
from run_all import options

HASHES = {
    'generate_blur_jpeg_s2_audited.py': '79a1e49a68821e50fecfd8caa0c6627e5a840f276600ab75dd592bd5fdc0cffb',
    'generate_lr_only.py': '6cdc66a712aac431af4d2f79da6bd196e6a240b7964283688830b136834aa4f3',
}
S2_ID_SHA = '9e406032b7545b686f17e3e95ad56638977d758cf27c97f85119bdc8e8291eba'
S2_PATH_SHA = '3f4cc41579732199c17ee6327ccae7d7740aeab5c9030b28e7d53f324d0828f0'
BLUR_COLUMNS = ('source', 'condition', 'parameter', 'width', 'height', 'out_path',
                'pillow_version', 'source_sha256', 'output_sha256', 'source_exif_orientation')
LR_COLUMNS = ('source', 'scale', 'clean_width', 'clean_height', 'lr_width', 'lr_height',
              'lr_path', 'pillow_version')


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def digest_lines(lines):
    return sha256(('\n'.join(lines)+'\n').encode('utf-8')).hexdigest()


def checked_child(root, value):
    relative = Path(value)
    full = (root/relative).resolve()
    require(not relative.is_absolute() and '..' not in relative.parts and full.is_relative_to(root),
            f'Unsafe relative path {value}')
    return full


def manifest(path, columns):
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == columns, f'Unexpected columns in {path}')
        return list(reader)


def encoded(image, fmt, **kwargs):
    stream = BytesIO()
    image.save(stream, format=fmt, **kwargs)
    return stream.getvalue()


def verify_variants(blur_dir, lr_dir, clean, hashes, sigmas, qualities):
    wanted = {(name, kind, value) for name in clean for kind, values in
              [('blur', sigmas), ('jpeg', qualities)] for value in values}
    seen = set()
    files = set()
    preview_paths = {}
    for row in manifest(blur_dir/'manifest.csv', BLUR_COLUMNS):
        name, kind, value = row['source'], row['condition'], float(row['parameter'])
        key = name, kind, value
        require(key in wanted and key not in seen, f'Unexpected/duplicate Blur/JPEG key {key}')
        seen.add(key)
        image = clean[name]
        require((int(row['width']), int(row['height'])) == image.size, f'Canvas changed for {key}')
        require(row['pillow_version'] == '11.3.0' and row['source_exif_orientation'] == '1', f'EXIF/Pillow mismatch {key}')
        require(row['source_sha256'] == hashes[name], f'Source hash mismatch {key}')
        output = checked_child(blur_dir, row['out_path'])
        require(output not in files and digest(output) == row['output_sha256'], f'Output path/hash mismatch {key}')
        files.add(output)
        preview_paths[key] = output
        with Image.open(output) as derived:
            require(derived.size == image.size and int(derived.getexif().get(274, 1)) == 1,
                    f'Output canvas/orientation mismatch {key}')
        reference = (encoded(image.filter(ImageFilter.GaussianBlur(radius=value)), 'PNG') if kind == 'blur'
                     else encoded(image, 'JPEG', quality=int(value)))
        require(sha256(reference).hexdigest() == row['output_sha256'], f'RGB no-rotation reference mismatch {key}')
    require(seen == wanted, 'Missing Blur/JPEG variants')

    wanted_lr = {(name, factor) for name in clean for factor in (2, 4, 8)}
    seen_lr, files_lr = set(), set()
    for row in manifest(lr_dir/'manifest.csv', LR_COLUMNS):
        name, factor = row['source'], int(row['scale'])
        key = name, factor
        require(key in wanted_lr and key not in seen_lr, f'Unexpected/duplicate LR key {key}')
        seen_lr.add(key)
        image = clean[name]
        dimensions = image.width//factor, image.height//factor
        require(row['pillow_version'] == '11.3.0', f'Pillow mismatch {key}')
        require((int(row['clean_width']), int(row['clean_height'])) == image.size and
                (int(row['lr_width']), int(row['lr_height'])) == dimensions, f'LR dimensions mismatch {key}')
        output = checked_child(lr_dir, row['lr_path'])
        require(output not in files_lr, f'Duplicate LR output path {key}')
        files_lr.add(output)
        with Image.open(output) as derived:
            require(derived.size == dimensions and derived.format == 'PNG' and
                    int(derived.getexif().get(274, 1)) == 1, f'LR geometry/format mismatch {key}')
        reference = encoded(image.resize(dimensions, Image.Resampling.BICUBIC), 'PNG')
        require(digest(output) == sha256(reference).hexdigest(), f'Bicubic no-rotation reference mismatch {key}')
    require(seen_lr == wanted_lr, 'Missing LR variants')
    return preview_paths


def preview(output, clean, paths, sigmas, qualities):
    sheet = Image.new('RGB', (1100, 190*len(clean)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row_index, (name, image) in enumerate(clean.items()):
        conditions = [('blur', value) for value in sigmas] + [('jpeg', value) for value in qualities]
        tiles = [('Clean', image)]
        for kind, value in conditions:
            with Image.open(paths[name, kind, value]) as derived:
                tiles.append((f'{kind} {value:g}', derived.convert('RGB')))
        for col, (label, tile) in enumerate(tiles):
            x, y = col*220, row_index*190
            draw.text((x+5, y+3), f'{Path(name).stem} {label}', fill='black')
            thumb = ImageOps.contain(tile, (210, 157))
            sheet.paste(thumb, (x+5, y+24))
    sheet.save(output, format='PNG')


def run(args, work):
    require(PIL.__version__ == '11.3.0', 'Require Pillow 11.3.0')
    audit = json.loads(args.audit_json.read_bytes())
    require(digest_lines(audit['s2_ids']) == audit['s2_ids_sha256'] == S2_ID_SHA and
            digest_lines(audit['s2_paths']) == audit['s2_paths_sha256'] == S2_PATH_SHA, 'Unexpected S2 lock')
    require(not audit['s1']['exif_not_1'] and not audit['s2']['exif_not_1'] and
            not audit['gt']['round_trip_failures'], 'Split/EXIF/round-trip must pass before dry-run')
    source_dir = args.generator_dir.resolve(strict=True)
    generator_hashes = {name: digest(source_dir/name) for name in HASHES}
    for name, expected in HASHES.items():
        require(generator_hashes[name] == expected, f'Wrong SHA-256 for {name}: {generator_hashes[name]}')
    sigmas = [float(value) for value in audit['sigma_cli']]
    qualities = audit['jpeg_qualities']
    require(audit['sigma_cli'] == ['1.88', '3.75'] and qualities == [30, 10], 'Unexpected frozen blur/JPEG parameters')
    clean_root = options([]).clean_root.resolve(strict=True)
    picks = sorted(audit['s1']['images'], key=lambda row: row[0])
    require(len(picks) == 500, 'Require all 500 verified S1 source records')
    first = next((row for row in picks if row[3] % 8 or row[4] % 8), picks[0])
    second = next(row for row in picks if row[0] != first[0])
    clean, hashes = {}, {}
    dry_input = work/'clean_s1_only'
    dry_input.mkdir()
    for source, name, expected_hash, width, height, orientation in (first, second):
        require(source in audit['s1_ids'] and source not in audit['s2_ids'] and name not in audit['s2_paths'],
                f'Expected S1-only dry-run source {source}')
        original = checked_child(clean_root, name)
        require(digest(original) == expected_hash and orientation == 1, f'Unexpected source hash/EXIF {name}')
        with Image.open(original) as image:
            require(image.size == (width, height) and int(image.getexif().get(274, 1)) == 1,
                    f'Unexpected source canvas/EXIF {name}')
            clean[name] = image.convert('RGB')
        hashes[name] = expected_hash
        local = checked_child(dry_input, name)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, local)
        require(digest(local) == expected_hash, f'Staged source hash mismatch {name}')
    source_list = work/'dry_run_sources.txt'
    source_list.write_bytes(('\n'.join(sorted(clean))+'\n').encode('utf-8'))
    blur_dir, lr_dir = work/'blur_jpeg', work/'native_lr'
    subprocess.run([sys.executable, str(source_dir/'generate_blur_jpeg_s2_audited.py'),
                    '--input-dir', str(dry_input), '--source-list', str(source_list), '--output-dir', str(blur_dir),
                    '--blur-sigmas', *audit['sigma_cli'], '--jpeg-qualities', '30', '10',
                    '--expect-count', str(len(clean)), '--strict-exif'], check=True)
    subprocess.run([sys.executable, str(source_dir/'generate_lr_only.py'),
                    '--input-dir', str(dry_input), '--output-dir', str(lr_dir)], check=True)
    paths = verify_variants(blur_dir, lr_dir, clean, hashes, sigmas, qualities)
    preview(work/'orientation_preview.png', clean, paths, sigmas, qualities)
    return dict(status='PASS_GENERATOR_DRY_RUN_ONLY', pillow=PIL.__version__, python=sys.version.split()[0],
                sha256_on_hpc=generator_hashes, audit_json_sha256=digest(args.audit_json),
                dry_run_source_ids=[first[0], second[0]], dry_run_source_sha256=hashes,
                blur_jpeg_rows=4*len(clean), lr_rows=3*len(clean), blur_sigmas=audit['sigma_cli'],
                jpeg_qualities=qualities, preview='orientation_preview.png',
                human_preview_review='PENDING', full_task1_go=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--generator-dir', type=Path,
                        default=Path(__file__).resolve().parents[1]/'.runtime/rev4_task1/handoff')
    args = parser.parse_args()
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    try:
        receipt = run(args, work)
    except (OSError, ValueError, KeyError, StopIteration, subprocess.CalledProcessError) as exc:
        receipt = dict(status='FAIL', detail=str(exc), full_task1_go=False)
    (work/'generator_dry_run.json').write_bytes(
        (json.dumps(receipt, sort_keys=True, allow_nan=False)+'\n').encode('utf-8'))
    print(receipt['status'], work/'generator_dry_run.json')
    if receipt['status'] == 'FAIL':
        print(receipt['detail'], file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
