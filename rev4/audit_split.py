"""Read-only Rev.4 S1/S2 selection and source-image audit on UIT-HPC."""
import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
from shapely import get_coordinates, intersection, union
from shapely.affinity import scale
from shapely.errors import GEOSException
from shapely.geometry import shape

from run_all import options

ROUND_TRIP_MIN_IOU = 1 - 1e-9
VERTEX_TOLERANCE_PX = 1e-9
ROUND_TRIP_GRID_PX = 1e-9


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical_hash(lines):
    return hashlib.sha256(('\n'.join(lines) + '\n').encode('utf-8')).hexdigest()


def distribution(values, percentiles):
    return np.quantile(values, percentiles, method='linear').tolist()


def recheck_round_trip_on_grid(clean, restored):
    """Only recheck valid, numerically equivalent polygon coordinates."""
    before = get_coordinates(clean)
    after = get_coordinates(restored)
    evidence = dict(max_vertex_error_px=None, grid_iou=None, accepted=False)
    if before.shape == after.shape and len(before):
        errors = np.linalg.norm(before-after, axis=1)
        if np.isfinite(errors).all():
            evidence['max_vertex_error_px'] = float(errors.max())
    vertex_error = evidence['max_vertex_error_px']
    if (vertex_error is None or vertex_error > VERTEX_TOLERANCE_PX or
            not clean.is_valid or not restored.is_valid or
            not clean.equals_exact(restored, VERTEX_TOLERANCE_PX)):
        evidence['reason'] = 'validity, structure or vertex tolerance check failed'
        return evidence
    try:
        intersection_area = intersection(clean, restored, grid_size=ROUND_TRIP_GRID_PX).area
        union_area = union(clean, restored, grid_size=ROUND_TRIP_GRID_PX).area
    except GEOSException as exc:
        evidence['reason'] = str(exc)
        return evidence
    if not math.isfinite(intersection_area) or not math.isfinite(union_area) or union_area <= 0:
        evidence['reason'] = 'invalid fixed-grid area'
        return evidence
    evidence['grid_iou'] = intersection_area / union_area
    evidence['accepted'] = evidence['grid_iou'] >= ROUND_TRIP_MIN_IOU
    evidence['reason'] = 'fixed-grid IoU passed' if evidence['accepted'] else 'fixed-grid IoU below tolerance'
    return evidence


def group(rows, ids, image_root):
    root = image_root.resolve(strict=True)
    exceptional, sizes, valid, ignored, image_rows = [], [], 0, 0, []
    for source in ids:
        row = rows[source]
        relative = Path(row['clean_relpath'])
        image_path = (root/relative).resolve(strict=True)
        assert not relative.is_absolute() and image_path.is_relative_to(root)
        digest = sha(image_path)
        assert digest == row['clean_sha256'], (source, relative, digest)
        with Image.open(image_path) as image:
            width, height = image.size
            orientation = int(image.getexif().get(274, 1))
        assert [width, height] == row['clean_size'], (source, relative, image.size)
        if orientation != 1:
            exceptional.append(dict(source_id=source, path=relative.as_posix(), orientation=orientation))
        sizes.append((width, height))
        valid += len(row['gt_clean']['valid'])
        ignored += len(row['gt_clean']['ignored'])
        image_rows.append([source, relative.as_posix(), digest, width, height, orientation])
    return dict(sources=len(ids), valid_gt=valid, ignored_gt=ignored, exif_not_1=exceptional,
                image_sizes={key: distribution(vals, [0, .5, 1]) for key, vals in
                             [('width', [w for w, _ in sizes]), ('height', [h for _, h in sizes]),
                              ('long_side', [max(pair) for pair in sizes])]},
                images=image_rows)


def gt_geometry(rows, ids):
    height_values = []
    bins = {str(s): Counter({'lt4': 0, '4_to_lt8': 0, '8_to_lt16': 0, 'ge16': 0}) for s in (1, 2, 4, 8)}
    raw_min_iou = dict.fromkeys((2, 4, 8), 1.0)
    verified_min_iou = raw_min_iou.copy()
    failures, rechecks = [], []
    boundaries = ('lt4', '4_to_lt8', '8_to_lt16', 'ge16')
    for source in ids:
        row = rows[source]
        w, h = row['clean_size']
        factors = {s: ((w//s)/w, (h//s)/h) for s in (1, 2, 4, 8)}
        assert all(x > 0 and y > 0 for x, y in factors.values()), source
        for entry in row['gt_clean']['valid']:
            geometry = shape(entry['geometry'])
            assert geometry.is_valid and geometry.area > 0, (source, entry['ann_id'])
            y0, y1 = geometry.bounds[1::2]
            height = y1 - y0
            height_values.append(height)
            for s, (sx, sy) in factors.items():
                native_h = height * sy
                bins[str(s)][boundaries[sum(native_h >= k for k in (4, 8, 16))]] += 1
                if s == 1:
                    continue
                native = scale(geometry, xfact=sx, yfact=sy, origin=(0, 0))
                restored = scale(native, xfact=1/sx, yfact=1/sy, origin=(0, 0))
                inter = geometry.intersection(restored).area
                raw_iou = inter / (geometry.area + restored.area - inter)
                verified_iou = raw_iou
                raw_min_iou[s] = min(raw_min_iou[s], raw_iou)
                if raw_iou < ROUND_TRIP_MIN_IOU:
                    evidence = dict(source_id=source, ann_id=entry['ann_id'], scale=s, raw_iou=raw_iou)
                    evidence.update(recheck_round_trip_on_grid(geometry, restored))
                    rechecks.append(evidence)
                    if evidence['accepted']:
                        verified_iou = evidence['grid_iou']
                    else:
                        failures.append(dict(evidence, iou=raw_iou))
                verified_min_iou[s] = min(verified_min_iou[s], verified_iou)
    height_stats = distribution(height_values, [0, .25, .5, .75, 1])
    return dict(height_n=len(height_values), height_min_q25_median_q75_max=height_stats,
                native_height_bins=bins, round_trip_min_iou=raw_min_iou,
                round_trip_max_error={s: 1-iou for s, iou in raw_min_iou.items()},
                round_trip_verified_min_iou=verified_min_iou, round_trip_failures=failures,
                round_trip_rechecks=rechecks,
                round_trip_recheck_policy=dict(min_iou=ROUND_TRIP_MIN_IOU,
                                               vertex_tolerance_px=VERTEX_TOLERANCE_PX,
                                               grid_px=ROUND_TRIP_GRID_PX))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='New directory for the audit and canonical source lists')
    cli = parser.parse_args()
    assert PIL.__version__ == '11.3.0'
    args = options([])
    split_path = args.output/'frozen_v2_0/split_lock.json'
    split = json.loads(split_path.read_bytes())
    manifest_digest = sha(args.validation_manifest)
    assert manifest_digest == split['validation_manifest_sha256']
    assert manifest_digest == '5cbf2230b0f83ab30816aca8afc5cb466e0b3301030ccd25d7d6d62d3136dfff'
    rows = {}
    for line in args.validation_manifest.read_bytes().splitlines():
        row = json.loads(line)
        source = row['source_image_id']
        assert source not in rows
        rows[source] = row
    assert len(rows) == 3262 and len({r['clean_sha256'] for r in rows.values()}) == 3262
    assert len({r['clean_relpath'] for r in rows.values()}) == 3262
    s1 = split['diagnostic']
    remaining = split['remaining_validation']
    assert split['seed'] == 20261107 and len(s1) == len(set(s1)) == 500
    assert len(remaining) == len(set(remaining)) == 2762
    assert not set(s1) & set(remaining) and set(s1) | set(remaining) == set(rows)
    assert s1 == sorted(random.Random(20261107).sample(sorted(rows), 500))
    s2 = sorted(random.Random(20261108).sample(sorted(set(remaining)), 500))
    assert len(s2) == len(set(s2)) == 500 and not set(s1) & set(s2)
    assert canonical_hash(s2) == '9e406032b7545b686f17e3e95ad56638977d758cf27c97f85119bdc8e8291eba'
    assert not {rows[s]['clean_sha256'] for s in s1} & {rows[s]['clean_sha256'] for s in s2}
    paths = [rows[s]['clean_relpath'] for s in s2]
    assert canonical_hash(paths) == '3f4cc41579732199c17ee6327ccae7d7740aeab5c9030b28e7d53f324d0828f0'
    after = sorted(set(remaining) - set(s2))
    assert len(after) == 2262
    s1_audit = group(rows, s1, args.clean_root)
    s2_audit = group(rows, s2, args.clean_root)
    gt_audit = gt_geometry(rows, s2)
    assert gt_audit['height_n'] == s2_audit['valid_gt']
    median = gt_audit['height_min_q25_median_q75_max'][2]
    raw = [.1*median, .2*median]
    sigmas = [round(x, 2) for x in raw]
    assert all(math.isfinite(x) and x > 0 for x in sigmas) and sigmas[0] != sigmas[1]
    result = dict(
        python=sys.version.split()[0], pillow=PIL.__version__, quantile_method='linear',
        selection_seed=20261108, manifest=dict(path=str(args.validation_manifest), sha256=manifest_digest,
                                             validation_sources=len(rows)),
        s1_split_lock=dict(path=str(split_path), sha256=sha(split_path)),
        s1_ids=s1, s1_ids_sha256=canonical_hash(s1), s1=s1_audit,
        remaining_before_s2=len(remaining), remaining_after_s2=len(after),
        s2_ids=s2, s2_ids_sha256=canonical_hash(s2), s2_paths=paths, s2_paths_sha256=canonical_hash(paths),
        s2=s2_audit, gt=gt_audit, sigma_raw=raw, sigma_cli=[format(x, '.2f') for x in sigmas],
        jpeg_qualities=[30, 10],
    )
    payload = json.dumps(result, sort_keys=True, allow_nan=False) + '\n'
    if cli.output_dir is None:
        print(payload, end='')
        return 0
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    (cli.output_dir/'s2_audit.json').write_bytes(payload.encode('utf-8'))
    for filename, lines in (('S1_source_ids.txt', s1), ('S2_source_ids.txt', s2), ('S2_sources.txt', paths)):
        (cli.output_dir/filename).write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    exif_failed = s1_audit['exif_not_1'] or s2_audit['exif_not_1']
    geometry_failed = gt_audit['round_trip_failures']
    status = ('BLOCKED — see s2_audit.json for EXIF or round-trip failures.'
              if exif_failed or geometry_failed else
              'PASS — split, EXIF and round-trip checks only. Full TASK 1 receipt is required for S2-A.')
    (cli.output_dir/'audit_status.txt').write_bytes((status + '\n').encode('utf-8'))
    for item in gt_audit['round_trip_rechecks']:
        print(f'{item["ann_id"]} LR{item["scale"]} raw IoU={item["raw_iou"]}, '
              f'grid IoU={item["grid_iou"]}, vertex error={item["max_vertex_error_px"]} px, {item["reason"]}')
    print(status)
    return 2 if exif_failed or geometry_failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
