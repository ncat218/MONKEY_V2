"""Reconstruct the frozen FULL_R1 train fold and prepare Clean/LR2 data."""
from collections import Counter
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lowres.common import canonical, jsonl, save_json, sha256, write_jsonl
from rev4 import require
from rev4.stage_a import make_task, rows_for, safe_path
from rev4.ssim_calibration import TRAIN_NAME, TRAIN_SHA256

LEGACY_COMMIT = '816c6d6aa0cd15de50a13c3ddfa621aed872b623'
TRAIN_SOURCES = 15_222
VAL_SOURCES = 2_262
RECIPE = dict(seed=42, optimizer_updates=10_000, micro_batch_size=2, accumulation=2,
              learning_rate=2e-5, weight_decay=1e-4, clip_grad=5.0,
              warmup_optimizer_updates=100, checkpoint_every_updates=2_500,
              validation_candidates=[5_000, 10_000], checkpoint_selection='max mean native clean/LR2 Hmean, earlier tie',
              clean_replay=0.5, precision='fp32', pillow='11.3.0',
              calibration=dict(mask_thr=[.30, .20, .15, .10, .05], min_text_score=[.30, .20, .40],
                               selection='max mean clean/LR2 Hmean, native-first tie'))


def balanced_indices(count, seed):
    require(count > 0, 'Need nonempty clean and LR2 groups')
    rng = random.Random(seed)
    while True:
        clean = list(range(count))
        degraded = list(range(count, 2*count))
        rng.shuffle(clean)
        rng.shuffle(degraded)
        for pair in zip(clean, degraded):
            yield from pair


def lr2_dimensions(size):
    return tuple(max(1, side//2) for side in size)


def native_lr2(image):
    import PIL
    from PIL import Image

    require(PIL.__version__ == RECIPE['pillow'], 'Native LR2 requires Pillow 11.3.0')
    require(int(image.getexif().get(274, 1)) == 1, 'Unexpected source EXIF orientation')
    return image.convert('RGB').resize(lr2_dimensions(image.size), Image.Resampling.BICUBIC)


def scale_instance(instance, clean_size, lr_size):
    sx, sy = (after/before for before, after in zip(clean_size, lr_size))
    factors = (sx, sy)
    return dict(instance,
                bbox=[value*factors[i % 2] for i, value in enumerate(instance['bbox'])],
                polygon=[value*factors[i % 2] for i, value in enumerate(instance['polygon'])])


def read_legacy_modules(repo, temporary):
    archive = temporary/'legacy.tar'
    with archive.open('wb') as sink:
        subprocess.run(['git', '-C', str(repo), 'archive', LEGACY_COMMIT, 'src/monkey_textdet'],
                       stdout=sink, check=True)
    with tarfile.open(archive) as source:
        source.extractall(temporary, filter='data')
    sys.path.insert(0, str(temporary/'src'))
    from monkey_textdet import data, full_data
    from monkey_textdet.vendor import shared_manifest
    return data, full_data, shared_manifest


def recover_train(a, destination, scratch, legacy, full_data, manifest):
    """Recreate membership from frozen metadata and open pixels only for the chosen TRAIN rows."""
    archived = a.validation_manifest.with_name(TRAIN_NAME)
    if archived.is_file():
        require(sha256(archived) == TRAIN_SHA256, 'Historical TRAIN SHA-256 mismatch')
        shutil.copyfile(archived, destination)
        return 'existing frozen TRAIN'

    records, provenance = full_data.read_metadata(a.raw_sources)
    from monkey_textdet.common import read_json

    official_ids = {str(item['image_id']) for item in read_json(a.raw_sources['clean_json'])['data']}
    raw = read_json(a.raw_sources['textocr_json'])
    textocr, _ = manifest.normalize_textocr(raw)
    del raw
    memberships = []
    counts = Counter()
    for row in sorted(records, key=lambda item: item['image_id']):
        source = row['image_id']
        if row['split'] != 'train':
            continue
        require(source in official_ids, f'FULL_R1 train metadata outside official train {source}')
        if source not in textocr['imgs'] or not textocr['img2Anns'].get(source):
            continue
        size = int(row['clean_width']), int(row['clean_height'])
        degraded = int(row['degraded_width']), int(row['degraded_height'])
        _, roi, _ = manifest.source_geometry(row, size, degraded)
        gt, _ = manifest.gt_for_source(source, textocr, roi, manifest.DEFAULT_SHARED_CONFIG, counts)
        if gt['valid']:
            memberships.append(dict(source_image_id=source, clean_sha256=row['source_image_sha256']))
    del textocr
    require(len(memberships) == 21_747, 'FULL_R1 membership count differs from the frozen split')
    folds = legacy.split_source_groups(memberships, dict(train=.7, validation=.15, test=.15), 42)
    frozen_val = {row['source_image_id']: row['clean_sha256'] for row in jsonl(a.validation_manifest)}
    by_source = {row['source_image_id']: row['clean_sha256'] for row in memberships}
    require(len(folds['train']) == TRAIN_SOURCES and len(frozen_val) == 3262 and
            {source: by_source[source] for source in folds['validation']} == frozen_val,
            'Reconstructed membership differs from frozen FULL_R1 validation')
    selected = set(folds['train'])
    train_records = [row for row in records if row['image_id'] in selected]
    require(len(train_records) == TRAIN_SOURCES, 'Reconstructed TRAIN membership count differs')
    print(f'Locked membership recovered. Verifying pixels for {len(train_records)} TRAIN sources.', flush=True)
    with patch.object(full_data, 'read_metadata', return_value=(train_records, provenance)):
        rebuilt = full_data.build_full_shared(a.raw_sources, scratch/'train_only_recovery')
    rebuilt_path = Path(rebuilt['manifest'])
    require(rebuilt['sources'] == TRAIN_SOURCES and sha256(rebuilt_path) == TRAIN_SHA256,
            f'Rebuilt TRAIN SHA-256 mismatch: {sha256(rebuilt_path)}')
    shutil.move(rebuilt_path, destination)
    return f'exact TRAIN rebuilt with {LEGACY_COMMIT}'


def check_split(train_file, validation, s1, s2):
    train_ids, train_hashes = set(), set()
    for row in jsonl(train_file):
        source = row['source_image_id']
        require(source not in train_ids, f'Duplicate TRAIN ID {source}')
        train_ids.add(source)
        train_hashes.add(row['clean_sha256'])
    val_ids = set(validation)
    val_hashes = {row['clean_sha256'] for row in validation.values()}
    require(len(train_ids) == TRAIN_SOURCES and not train_ids & val_ids and not train_hashes & val_hashes,
            'TRAIN/validation identity or clean-SHA overlap')
    require(len(s1) == len(s2) == 500 and not set(s1) & set(s2) and
            (set(s1) | set(s2)) <= val_ids, 'Invalid locked pilot cohorts')
    remaining = sorted(val_ids - set(s1) - set(s2))
    require(len(remaining) == VAL_SOURCES, 'Expected 2,262 remaining validation sources')
    return remaining


def verify_clean(root, row):
    from PIL import Image

    path = safe_path(root, row['clean_relpath'])
    require(sha256(path) == row['clean_sha256'], f'Unexpected clean bytes {row["source_image_id"]}')
    with Image.open(path) as image:
        require(list(image.size) == row['clean_size'] and int(image.getexif().get(274, 1)) == 1,
                f'Unexpected clean dimensions or EXIF {row["source_image_id"]}')
    return path


def train_annotations(train, root, destination, legacy):
    """Write two equal groups. Only training geometry is adapted to MMOCR polygons."""
    audit = Counter()
    counts = Counter()
    metainfo = dict(dataset_type='TextDetDataset', task_name='textdet', category=[dict(id=0, name='text')])
    with destination.open('wb') as sink:
        sink.write(b'{"metainfo":'+canonical(metainfo).rstrip(b'\n')+b',"data_list":[')
        separator = b''
        for condition in ('clean', 'lr2'):
            for row in jsonl(train):
                source = row['source_image_id']
                clean_size = row['clean_size']
                path = (verify_clean(root, row) if condition == 'clean' else
                        safe_path(root, row['clean_relpath']))
                instances = legacy.instances_for(row, 'clean', audit)
                size = clean_size
                if condition == 'lr2':
                    size = lr2_dimensions(clean_size)
                    instances = [scale_instance(item, clean_size, size) for item in instances]
                record = dict(img_path=str(path), width=size[0], height=size[1], condition=condition,
                              sample_id=f'{condition}::{source}', instances=instances)
                sink.write(separator+canonical(record).rstrip(b'\n'))
                separator = b',\n'
                counts[condition] += 1
                counts[f'{condition}_valid_instances'] += sum(not item['ignore'] for item in instances)
        sink.write(b']}\n')
    require(counts['clean'] == counts['lr2'] == TRAIN_SOURCES, 'Expected equal TRAIN groups')
    return dict(counts=dict(counts), target_conversion=dict(audit))


def validation_tasks(root, rows, remaining, stage, final):
    from PIL import Image

    tasks = []
    for source in remaining:
        row = rows[source]
        path = verify_clean(root, row)
        tasks.append(make_task(source, 'clean', path, row['clean_size'], row['clean_sha256']))
        with Image.open(path) as image:
            lr2 = native_lr2(image)
        # Preserve the lossless LR2 pixels while leaving ICC metadata to the raw source.
        lr2.info.clear()
        generated = stage/'validation_lr2'/f'{source}.png'
        generated.parent.mkdir(parents=True, exist_ok=True)
        lr2.save(generated, format='PNG')
        tasks.append(make_task(source, 'lr2', final/generated.relative_to(stage), lr2.size, sha256(generated)))
    return tasks


def prepare(a):
    import PIL

    require(PIL.__version__ == RECIPE['pillow'], 'B1 requires Pillow 11.3.0')
    final = a.output/'prepared'
    require(not final.exists(), f'Unexpected incomplete preparation directory {final}')
    with TemporaryDirectory(prefix='.preparing-', dir=a.output) as temp:
        scratch = Path(temp)
        stage = scratch/'prepared'
        stage.mkdir()
        legacy, full_data, shared_manifest = read_legacy_modules(a.legacy_repo, scratch)
        train = stage/TRAIN_NAME
        restoration = recover_train(a, train, scratch, legacy, full_data, shared_manifest)
        require(sha256(train) == TRAIN_SHA256, 'Prepared TRAIN SHA-256 mismatch')
        validation_ids = sorted(set(a.s1_ids) | set(a.s2_ids) | set(a.original_remaining_ids))
        validation = rows_for(a, validation_ids)
        remaining = check_split(train, validation, a.s1_ids, a.s2_ids)
        write_jsonl(stage/'validation_remaining.jsonl', (validation[source] for source in remaining))
        root = a.clean_root.resolve(strict=True)
        print('Preparing clean and LR2 TRAIN targets', flush=True)
        audit = train_annotations(train, root, stage/'train_annotations.json', legacy)
        print('Preparing clean and LR2 remaining-validation inputs', flush=True)
        tasks = validation_tasks(root, validation, remaining, stage, final)
        write_jsonl(stage/'validation_tasks.jsonl', tasks)
        require(len(tasks) == 2*VAL_SOURCES, 'Unexpected number of remaining-validation tasks')
        file_names = (TRAIN_NAME, 'train_annotations.json', 'validation_remaining.jsonl', 'validation_tasks.jsonl')
        save_json(stage/'prepared.json', dict(train_restoration=restoration, train_sources=TRAIN_SOURCES,
                  remaining_validation_sources=VAL_SOURCES, generator_sha256=a.lock['lr_generator_sha256'],
                  file_sha256={name: sha256(stage/name) for name in file_names}, audit=audit))
        stage.rename(final)
    print(f'B1 inputs prepared and locked: {final}', flush=True)
