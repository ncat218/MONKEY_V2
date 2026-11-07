"""Freeze validation-only inputs, LR variants and the clean E0 identity reference."""
import csv
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from shapely.affinity import scale as scale_geometry
from shapely.geometry import shape

from . import evaluate_shared as evaluator
from .common import jsonl, read_json, run_lock, save_json, sha256, write_jsonl

from .generate_lr_only import MANIFEST_FIELDS, bicubic, generate_variants, require_pillow

VALIDATION_SHA = '5cbf2230b0f83ab30816aca8afc5cb466e0b3301030ccd25d7d6d62d3136dfff'
PROTOCOL_SHA = 'b20bc96ac675fbc6b7db4c03e6dd1fc66e8661ff2474772bb44498a5e3763958'
EVALUATOR_SHA = 'ea25c8d9fc240ec10dc6eb45883b163c8e92952ee885423985e6d013f026fcdc'
E0_PREDICTIONS_SHA = '38dedf77b05c8da71dfb5ef2a5a0f353ca43ad3410621228903966ce3386e80a'
E0_CHECKPOINT_SHA = '4b13e26ad5c2e1b13fef45ac608298254d083062d2b9cafba706064bd8e2cf82'
E0_CONFIG_SHA = '94e4edd820b7cb8faca4a9414928af76f5a4f9cb225812fabad13eec74269899'
SEED = 20261107
NUM_VALIDATION = 3262
NUM_DIAGNOSTIC = 500
HEIGHT_BINS = ('lt4', '4_to_lt8', '8_to_lt16', 'ge16')


def checked_rows(args):
    files = (
        (args.validation_manifest, VALIDATION_SHA),
        (args.evaluation_protocol, PROTOCOL_SHA),
        (args.e0_validation_predictions, E0_PREDICTIONS_SHA),
        (evaluator.__file__, EVALUATOR_SHA),
    )
    for file, expected in files:
        if sha256(file) != expected:
            raise ValueError(f'Frozen FULL_R1 SHA-256 mismatch: {file}')
    rows = {row['source_image_id']: row for row in jsonl(args.validation_manifest)}
    if len(rows) != NUM_VALIDATION or len({r['clean_sha256'] for r in rows.values()}) != NUM_VALIDATION:
        raise ValueError('Expected 3262 unique validation source IDs and clean image hashes')
    diagnostic = sorted(random.Random(SEED).sample(sorted(rows), NUM_DIAGNOSTIC))
    remaining = sorted(rows.keys() - set(diagnostic))
    return rows, diagnostic, remaining


def clean_reference(args, selected):
    protocol = read_json(args.evaluation_protocol)
    reference, per_source = [], []
    totals = [0, 0, 0, 0]
    seen = set()
    for prediction in jsonl(args.e0_validation_predictions):
        source = prediction['source_image_id']
        if prediction['condition'] != 'clean' or source not in selected:
            continue
        row = selected[source]
        if source in seen or prediction['image_sha256'] != row['clean_sha256']:
            raise ValueError(f'E0 clean prediction identity mismatch: {source}')
        seen.add(source)
        reference.append(prediction)
        result = evaluator.evaluate_image(row['gt_clean'], row['roi_clean'], prediction, protocol)
        per_source.append(dict(source_image_id=source, counts=result['counts'],
                               matched_ann_ids=result['matched_ann_ids']))
        totals = [a + b for a, b in zip(totals, result['counts'])]
    if seen != selected.keys():
        raise ValueError('Missing historical E0 clean predictions for the diagnostic subset')
    tp, gt, pred, _ = totals
    summary = dict(sources=len(seen), tp=tp, fp=pred-tp, fn=gt-tp, **evaluator.rates(totals))
    return reference, per_source, summary


def check_geometry(geometries, clean_size, lr_size, height_counts):
    sx, sy = (lr / clean for lr, clean in zip(lr_size, clean_size))
    smallest_iou = 1.0
    for geom in geometries:
        native = scale_geometry(geom, xfact=sx, yfact=sy, origin=(0, 0))
        restored = scale_geometry(native, xfact=1 / sx, yfact=1 / sy, origin=(0, 0))
        intersection = geom.intersection(restored).area
        iou = intersection / (geom.area + restored.area - intersection)
        if iou < 1 - 1e-9:
            raise ValueError(f'Clean → LR → clean IoU below tolerance: {iou}')
        smallest_iou = min(smallest_iou, iou)
        _, y0, _, y1 = native.bounds
        height = y1 - y0
        height_counts[HEIGHT_BINS[sum(height >= boundary for boundary in (4, 8, 16))]] += 1
    return smallest_iou


def prepare(args, output):
    require_pillow()
    output.mkdir(parents=True, exist_ok=True)
    final = output/'frozen_v2_0'
    with run_lock(output):
        if final.exists():
            raise FileExistsError(f'Prepared pilot already exists: {final}')
        rows, diagnostic, remaining = checked_rows(args)
        selected = {source: rows[source] for source in diagnostic}
        clean_root = args.clean_root.resolve(strict=True)
        with TemporaryDirectory(prefix='.preparing-', dir=output) as temporary:
            stage = Path(temporary)/final.name
            stage.mkdir()
            tasks = []
            audit = {name: dict(valid_gt=0, ignored_gt=0, valid_height_bins=dict.fromkeys(HEIGHT_BINS, 0))
                     for name in ('clean', 'lr2', 'lr4', 'lr8')}
            min_iou = 1.0
            with (stage/'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(MANIFEST_FIELDS)
                for source, row in selected.items():
                    relative = Path(row['clean_relpath'])
                    path = (clean_root/relative).resolve(strict=True)
                    if not path.is_relative_to(clean_root) or sha256(path) != row['clean_sha256']:
                        raise ValueError(f'Clean image identity mismatch: {source}')
                    with Image.open(path) as image:
                        clean = image.convert('RGB')
                    if list(clean.size) != row['clean_size']:
                        raise ValueError(f'Clean image size mismatch: {source}')
                    gt = row['gt_clean']
                    geometries = [shape(entry['geometry']) for entry in gt['valid']]
                    for counts in audit.values():
                        counts['valid_gt'] += len(gt['valid'])
                        counts['ignored_gt'] += len(gt['ignored'])
                    tasks.append(dict(sample_id=f'clean::{source}', source_image_id=source, condition='clean',
                                      path=str(path), size=list(clean.size), image_sha256=row['clean_sha256']))
                    check_geometry(geometries, clean.size, clean.size, audit['clean']['valid_height_bins'])
                    for entry, lr in generate_variants(clean, relative, stage):
                        writer.writerow(entry)
                        scale, lr_path = entry[1], Path(entry[6])
                        condition = f'lr{scale}'
                        min_iou = min(min_iou, check_geometry(geometries, clean.size, lr.size,
                                                              audit[condition]['valid_height_bins']))
                        tasks.append(dict(sample_id=f'{condition}::{source}', source_image_id=source,
                                          condition=condition, path=str(final/lr_path), size=list(lr.size),
                                          image_sha256=sha256(stage/lr_path), clean_size=list(clean.size)))
                        if scale == 4:
                            up_path = Path('b_up_lr4')/relative.parent/lr_path.name
                            (stage/up_path).parent.mkdir(parents=True, exist_ok=True)
                            bicubic(lr, clean.size).save(stage/up_path, format='PNG')
                            tasks.append(dict(sample_id=f'b_up_lr4::{source}', source_image_id=source,
                                              condition='b_up_lr4', path=str(final/up_path), size=list(clean.size),
                                              image_sha256=sha256(stage/up_path)))
            reference, per_source, summary = clean_reference(args, selected)
            save_json(stage/'split_lock.json', dict(seed=SEED, algorithm='sample sorted source IDs with Python Random',
                      validation_manifest_sha256=VALIDATION_SHA, diagnostic=diagnostic, remaining_validation=remaining))
            write_jsonl(stage/'tasks.jsonl', tasks)
            write_jsonl(stage/'e0_clean_reference.jsonl', reference)
            write_jsonl(stage/'e0_clean_matched_counts.jsonl', per_source)
            save_json(stage/'preflight.json', dict(
                validation_sources=NUM_VALIDATION, diagnostic_sources=len(diagnostic),
                remaining_validation_sources=len(remaining), condition_audit=audit,
                min_clean_lr_clean_iou=min_iou, num_inference_tasks=len(tasks),
                validation_manifest_sha256=VALIDATION_SHA, evaluation_protocol_sha256=PROTOCOL_SHA,
                evaluator_sha256=EVALUATOR_SHA, e0_predictions_sha256=E0_PREDICTIONS_SHA,
                e0_checkpoint_sha256=E0_CHECKPOINT_SHA, e0_config_sha256=E0_CONFIG_SHA,
                e0_clean_reference=summary))
            stage.rename(final)
    return final
