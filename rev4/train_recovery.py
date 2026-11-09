"""Recover the locked FULL_R1 TRAIN manifest from the surviving job 8323 audit."""
from collections import Counter
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from lowres.common import canonical, jsonl, read_json, save_json, sha256
from rev4 import require
from rev4.training_data import (LEGACY_COMMIT, TRAIN_NAME, TRAIN_SHA256, TRAIN_SOURCES,
                               degraded_relative, read_legacy_modules)


AUDIT = Path('/datastore/dungmt/textvqa/Danh/results/nckh_restart_v1/phase_a_full_textvqa_seed42'
             '/phase_a_unified_manifest.csv')
AUDIT_SHA256 = 'a2f03bb1984aa93cc2d677a533cc1db690e6387c01e80c97c681b496f87116db'
CONFIG_SHA256 = '141df1cf4fefb34d380d1f240e32da702fbebe1067154b35ba1644d92f901a10'
QA_SHA256 = '081e7f2aba052309d1692d8b4055faee980b5f33bff522fd54ee0b07851afcf1'
TEXTVQA_TRAIN_SHA256 = '95f5c407db56cba56a177799dcd685a7cc0ec7c0d851b59910acf7786d31b68a'
TEXTOCR_TRAIN_SHA256 = '5bac8303c5999a3ddf630efde8ac9743b4461be6022b193a1075df90fbd2ce0c'


def audited_sources(path, clean_root, official_ids):
    sources = {}
    count = 0
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] != 'train':
                continue
            count += 1
            source = row['image_id']
            require(source == row['source_image_id'] and (row['condition'], row['level']) == ('realistic_mix', 'L2')
                    and row['config_sha256'] == CONFIG_SHA256 and row['source_manifest_sha256'] == QA_SHA256,
                    f'Historical generator identity differs for {source}')
            clean_path = Path(row['clean_image_path']).relative_to(clean_root)
            size = [int(row['clean_width']), int(row['clean_height'])]
            require('..' not in clean_path.parts and size == [int(row['degraded_width']), int(row['degraded_height'])],
                    f'Unexpected historical path or canvas for {source}')
            item = dict(source_image_id=source, clean_relpath=clean_path.as_posix(),
                        degraded_relpath=degraded_relative(row['degraded_image_path']).as_posix(),
                        clean_sha256=row['source_image_sha256'], degraded_sha256=row['degraded_sha256'],
                        size=size, params=json.loads(row['params']))
            require(sources.setdefault(source, item) == item, f'Conflicting metadata across QA rows for {source}')
    require(count == 34_602 and len(sources) == 21_953 and set(sources) == official_ids,
            'Job 8323 metadata differs from the audited official train inventory')
    return sources


def targets(manifest, textocr, row, audit):
    size = tuple(row['size'])
    matrix, roi, ops = manifest.source_geometry(row, size, size)
    gt, removed = manifest.gt_for_source(row['source_image_id'], textocr, roi,
                                        manifest.DEFAULT_SHARED_CONFIG, audit)
    return matrix, roi, ops, gt, removed


def frozen_row(full_data, manifest, textocr, row):
    """Serialize the original detector manifest schema from audited mixed metadata."""
    from shapely.geometry import mapping, shape

    source = row['source_image_id']
    matrix, roi, ops, gt, removed = targets(manifest, textocr, row, Counter())
    require(gt['valid'], f'Missing usable GT for {source}')
    mixed_gt = {key: [dict(item, geometry=mapping(manifest.warp_geometry(shape(item['geometry']), matrix)))
                      for item in gt[key]] for key in ('valid', 'ignored')}
    return dict(schema_version=manifest.SCHEMA_VERSION, source_image_id=source,
                variant_id=full_data.identity([source, row['degraded_sha256'], CONFIG_SHA256])[:24],
                official_split='train', clean_relpath=row['clean_relpath'], degraded_relpath=row['degraded_relpath'],
                clean_sha256=row['clean_sha256'], degraded_sha256=row['degraded_sha256'],
                clean_size=row['size'], degraded_size=row['size'], condition='realistic_mix', severity='L2',
                operations=ops, has_perspective='perspective' in ops, homography=matrix.tolist(),
                roi_clean=mapping(roi), roi_degraded=mapping(manifest.warp_geometry(roi, matrix)),
                gt_clean=gt, gt_degraded=mixed_gt, removed_gt_ids=removed,
                source_config_sha256=CONFIG_SHA256, severity_basis='image_relative_fallback')


def recover(a):
    for path, expected in ((AUDIT, AUDIT_SHA256), (a.raw_sources['clean_json'], TEXTVQA_TRAIN_SHA256),
                           (a.raw_sources['textocr_json'], TEXTOCR_TRAIN_SHA256)):
        require(sha256(path) == expected, f'Historical input SHA-256 differs for {path}')
    official_ids = {str(row['image_id']) for row in read_json(a.raw_sources['clean_json'])['data']}
    sources = audited_sources(AUDIT, a.clean_root, official_ids)
    validation = {row['source_image_id']: row for row in jsonl(a.validation_manifest)}
    require(len(validation) == 3262, 'Expected the frozen validation fold')
    with TemporaryDirectory(prefix='.recovering-', dir=a.recovery) as temporary:
        scratch = Path(temporary)
        legacy, full_data, manifest = read_legacy_modules(a.legacy_repo, scratch)
        textocr, _ = manifest.normalize_textocr(read_json(a.raw_sources['textocr_json']))
        for source, original in validation.items():
            recreated = frozen_row(full_data, manifest, textocr, sources[source])
            require(canonical(recreated) == canonical(original), f'Frozen validation row differs for {source}')
        print(f'All {len(validation)} frozen validation rows match exactly.', flush=True)
        audit = Counter()
        membership = []
        for source, row in sources.items():
            if source not in textocr['imgs'] or not textocr['img2Anns'].get(source):
                continue
            if source in validation or targets(manifest, textocr, row, audit)[3]['valid']:
                membership.append(dict(source_image_id=source, clean_sha256=row['clean_sha256']))
        require(len(membership) == 21_747, 'Recreated membership differs from the historical audit')
        folds = legacy.split_source_groups(membership, dict(train=.7, validation=.15, test=.15), 42)
        require(set(folds['validation']) == set(validation) and len(folds['train']) == TRAIN_SOURCES,
                'Recreated split differs from the frozen split')
        print(f'Locked membership verified. Writing {TRAIN_SOURCES} TRAIN rows.', flush=True)
        verified = scratch/'verified'
        verified.mkdir()
        destination = verified/TRAIN_NAME
        full_data.write_jsonl(destination, (frozen_row(full_data, manifest, textocr, sources[source])
                                            for source in folds['train']))
        digest = sha256(destination)
        require(digest == TRAIN_SHA256, f'Reconstructed TRAIN SHA-256 mismatch: {digest}')
        save_json(verified/'recovery.json', dict(status='PASS', train_sha256=digest, train_sources=TRAIN_SOURCES,
                  audit_job=8323, audit_sha256=AUDIT_SHA256, manifest_commit=LEGACY_COMMIT,
                  validation_rows=len(validation), validation_sha256=a.lock['validation_sha256']))
        verified.rename(a.recovery/'verified')
    print('PASS:', a.recovery/'verified'/TRAIN_NAME, flush=True)
