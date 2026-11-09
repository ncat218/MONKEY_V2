"""Recover the byte-identical FULL_R1 TRAIN manifest when both backups are missing."""
from collections import Counter
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from lowres.common import canonical, jsonl, read_json, save_json, sha256, write_jsonl
from rev4 import require
from rev4.full_r1_generator import CONFIG_SHA256, GENERATOR_COMMIT, OPERATORS_SHA256, QA_SHA256
from rev4.training_data import LEGACY_COMMIT, TRAIN_NAME, TRAIN_SHA256, TRAIN_SOURCES, read_legacy_modules


TEXTVQA_TRAIN_SHA256 = '95f5c407db56cba56a177799dcd685a7cc0ec7c0d851b59910acf7786d31b68a'
TEXTOCR_TRAIN_SHA256 = '5bac8303c5999a3ddf630efde8ac9743b4461be6022b193a1075df90fbd2ce0c'


def source_inventory(a, textocr):
    official = {str(row['image_id']) for row in read_json(a.raw_sources['clean_json'])['data']}
    require(len(official) == 21_953, 'Expected the audited official TextVQA train inventory')
    index = {}
    for path in (a.clean_root/'train_images').rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            source = path.stem.lower()
            require(source not in index, f'Ambiguous clean source filename {source}')
            index[source] = path
    candidates = {}
    for source in sorted(official):
        if source not in textocr['imgs'] or not textocr['img2Anns'].get(source):
            continue
        meta = textocr['imgs'][source]
        candidates[source] = dict(source_image_id=source, path=str(index[source.lower()]),
                                  size=[int(meta['width']), int(meta['height'])])
    require(len(candidates) == 21_778, 'TextOCR coverage differs from the historical audit')
    return candidates


def replay(a, scratch, mode, rows):
    inputs, outputs = scratch/'replay_inputs.jsonl', scratch/'replay_outputs.jsonl'
    write_jsonl(inputs, rows)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(a.recovery/'packages'))
    subprocess.run([str(a.python), '-u', '-m', 'rev4.full_r1_generator.replay', mode, str(inputs), str(outputs)],
                   check=True, env=env)
    return {row['source_image_id']: row for row in jsonl(outputs)}


def targets(manifest, textocr, row, params, audit):
    size = tuple(row['size'])
    matrix, roi, operations = manifest.source_geometry(dict(params=params), size, size)
    gt, removed = manifest.gt_for_source(row['source_image_id'], textocr, roi,
                                        manifest.DEFAULT_SHARED_CONFIG, audit)
    return matrix, roi, operations, gt, removed


def frozen_row(root, full_data, manifest, textocr, clean, mixed):
    """Historical FULL_R1 row schema and field order, serialized by legacy code."""
    from shapely.geometry import mapping, shape

    source = clean['source_image_id']
    matrix, roi, ops, gt, removed = targets(manifest, textocr, clean, mixed['params'], Counter())
    require(gt['valid'], f'Missing usable TRAIN GT for {source}')
    mixed_gt = {key: [dict(item, geometry=mapping(manifest.warp_geometry(shape(item['geometry']), matrix)))
                      for item in gt[key]] for key in ('valid', 'ignored')}
    mixed_sha = mixed['degraded_sha256']
    return dict(schema_version=manifest.SCHEMA_VERSION, source_image_id=source,
                variant_id=full_data.identity([source, mixed_sha, CONFIG_SHA256])[:24], official_split='train',
                clean_relpath=Path(clean['path']).relative_to(root).as_posix(),
                degraded_relpath=f'main/images/realistic_mix/L2/train/{source}.png',
                clean_sha256=clean['clean_sha256'], degraded_sha256=mixed_sha,
                clean_size=clean['size'], degraded_size=clean['size'], condition='realistic_mix', severity='L2',
                operations=ops, has_perspective='perspective' in ops, homography=matrix.tolist(),
                roi_clean=mapping(roi), roi_degraded=mapping(manifest.warp_geometry(roi, matrix)),
                gt_clean=gt, gt_degraded=mixed_gt, removed_gt_ids=removed,
                source_config_sha256=CONFIG_SHA256, severity_basis='image_relative_fallback')


def verify_sentinels(a, scratch, full_data, manifest, textocr, candidates, validation):
    sentinel_ids = {}
    for source, row in validation.items():
        require(row['source_config_sha256'] == CONFIG_SHA256, f'Unexpected historical generator for {source}')
        sentinel_ids.setdefault(tuple(row['operations']), source)
    examples = [dict(candidates[source], clean_sha256=validation[source]['clean_sha256'])
                for source in sentinel_ids.values()]
    mixed = replay(a, scratch, 'png', examples)
    for row in examples:
        source = row['source_image_id']
        frozen = validation[source]
        require(mixed[source]['degraded_sha256'] == frozen['degraded_sha256'],
                f'Original PNG SHA-256 mismatch for validation sentinel {source}; stop recovery')
        rebuilt = frozen_row(a.clean_root, full_data, manifest, textocr, row, mixed[source])
        different = [key for key, value in frozen.items() if canonical(value) != canonical(rebuilt[key])]
        require(not different, f'Legacy geometry differs for validation sentinel {source}: {different}')
    print(f'Original PNG and manifest rows match for {len(examples)} validation sentinels.', flush=True)
    return len(examples)


def recover(a):
    from rev4 import full_r1_generator

    operators = Path(full_r1_generator.__file__).with_name('operators.py')
    require(sha256(operators) == OPERATORS_SHA256, 'Original FULL_R1 operators source hash differs')
    for path, expected in ((a.raw_sources['clean_json'], TEXTVQA_TRAIN_SHA256),
                           (a.raw_sources['textocr_json'], TEXTOCR_TRAIN_SHA256)):
        require(sha256(path) == expected, f'Original annotation SHA-256 differs for {path}')
    validation = {row['source_image_id']: row for row in jsonl(a.validation_manifest)}
    require(len(validation) == 3262, 'Expected the frozen validation fold')
    with TemporaryDirectory(prefix='.recovering-', dir=a.recovery) as temporary:
        scratch = Path(temporary)
        legacy, full_data, manifest = read_legacy_modules(a.legacy_repo, scratch)
        textocr, _ = manifest.normalize_textocr(read_json(a.raw_sources['textocr_json']))
        candidates = source_inventory(a, textocr)
        sentinels = verify_sentinels(a, scratch, full_data, manifest, textocr, candidates, validation)
        metadata = replay(a, scratch, 'geometry', candidates.values())
        audit = Counter()
        membership = []
        for source, row in candidates.items():
            data = metadata[source]
            row['clean_sha256'] = data['clean_sha256']
            if source in validation:
                require(row['clean_sha256'] == validation[source]['clean_sha256'] and
                        row['size'] == validation[source]['clean_size'], f'Frozen validation bytes differ for {source}')
            elif not targets(manifest, textocr, row, data['params'], audit)[3]['valid']:
                continue
            membership.append(dict(source_image_id=source, clean_sha256=row['clean_sha256']))
        require(len(membership) == 21_747, 'Recreated membership differs from the historical audit')
        folds = legacy.split_source_groups(membership, dict(train=.7, validation=.15, test=.15), 42)
        require(set(folds['validation']) == set(validation) and len(folds['train']) == TRAIN_SOURCES,
                'Recreated split differs from the frozen split')
        selected = [candidates[source] for source in folds['train']]
        print(f'Locked membership matches validation. Replaying {len(selected)} TRAIN images in memory.', flush=True)
        mixed = replay(a, scratch, 'png', selected)
        verified = scratch/'verified'
        verified.mkdir()
        temporary_manifest = verified/TRAIN_NAME
        full_data.write_jsonl(temporary_manifest, (frozen_row(a.clean_root, full_data, manifest, textocr, row,
                                                              mixed[row['source_image_id']]) for row in selected))
        digest = sha256(temporary_manifest)
        require(digest == TRAIN_SHA256, f'Reconstructed TRAIN SHA-256 mismatch: {digest}')
        save_json(verified/'recovery.json', dict(status='PASS', train_sha256=digest, train_sources=TRAIN_SOURCES,
                  validation_sentinels=sentinels, generator_commit=GENERATOR_COMMIT, manifest_commit=LEGACY_COMMIT,
                  generator_config_sha256=CONFIG_SHA256, generator_qa_sha256=QA_SHA256,
                  validation_sha256=a.lock['validation_sha256']))
        verified.rename(a.recovery/'verified')
    print('PASS:', a.recovery/'verified'/TRAIN_NAME, flush=True)
