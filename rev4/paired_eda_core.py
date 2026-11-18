"""Instance-level audit around the unchanged frozen polygon evaluator."""
import math

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from lowres.evaluate_shared import evaluate_image, maximum_matching
from rev4 import require
from rev4.adaptation_plan import HEIGHT_CUTS, HEIGHT_NAMES


def filtered_geometry(prediction, row, policy):
    roi = shape(row['roi_clean'])
    ignore = unary_union([shape(g['geometry']) for g in row['gt_clean']['ignored']])
    shapes, invalid = {}, []
    require(len(prediction['polygons']) == len(prediction['scores']), 'Polygon/score lengths differ')
    for index, (coords, score) in enumerate(zip(prediction['polygons'], prediction['scores'])):
        require(math.isfinite(score) and 0 <= score <= 1, 'Invalid prediction score')
        points = np.asarray(coords, dtype=float)
        if points.size < 6 or points.size % 2 or not np.isfinite(points).all():
            invalid.append(index)
            continue
        polygon = Polygon(points.reshape(-1, 2))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            invalid.append(index)
            continue
        clipped = polygon.intersection(roi)
        if clipped.is_empty or clipped.area/polygon.area < policy['min_prediction_roi_fraction']:
            continue
        if not ignore.is_empty and clipped.intersection(ignore).area/clipped.area > policy['ignore_prediction_overlap']:
            continue
        shapes[index] = clipped
    return shapes, invalid


def overlap_index(geometries, scores):
    indices = list(geometries)
    shapes = list(geometries.values())
    return indices, shapes, scores, STRtree(shapes)


def overlaps(index, gt):
    indices, shapes, scores, tree = index
    found = []
    for local in tree.query(gt, predicate='intersects'):
        polygon = shapes[int(local)]
        inter = gt.intersection(polygon).area
        if inter:
            original = indices[int(local)]
            found.append((inter/(gt.area+polygon.area-inter), float(scores[original]), original))
    return sorted(found, key=lambda item: (-item[0], item[2]))


def trace_matching(prediction, row, policy):
    """Recover exact assigned pairs, asserting agreement with the official helper."""
    official = evaluate_image(row['gt_clean'], row['roi_clean'], prediction, policy)
    geometries, invalid = filtered_geometry(prediction, row, policy)
    index = overlap_index(geometries, prediction['scores'])
    originals = list(geometries)
    to_local = {original: i for i, original in enumerate(originals)}
    adjacency = [[to_local[p] for iou, _, p in overlaps(index, shape(gt['geometry'])) if iou >= policy['match_iou']]
                 for gt in row['gt_clean']['valid']]
    pairs = maximum_matching(adjacency, len(originals))
    assigned = {}
    for gt_index, local in pairs:
        gt = row['gt_clean']['valid'][gt_index]
        original = originals[local]
        geometry = shape(gt['geometry'])
        inter = geometry.intersection(geometries[original]).area
        assigned[str(gt['ann_id'])] = dict(prediction_index=original,
                        iou=inter/(geometry.area+geometries[original].area-inter),
                        score=float(prediction['scores'][original]))
    require(set(assigned) == set(map(str, official['matched_ann_ids'])), 'EDA matching changed the frozen GT assignments')
    used = {originals[p] for _, p in pairs}
    false_positives = [i for i in originals if i not in used]
    require(false_positives == official['fp_prediction_indices'] and len(invalid) == official['invalid_prediction_count'],
            'EDA FP filtering differs from the frozen evaluator')
    require(len(false_positives)+len(invalid) == official['counts'][2]-official['counts'][0], 'FP count mismatch')
    return official, assigned, geometries, false_positives+invalid


def gt_attributes(entry, clean_size):
    geometry = shape(entry['geometry'])
    x0, y0, x1, y1 = geometry.bounds
    h_lr2 = (y1-y0)*(max(1, clean_size[1]//2)/clean_size[1])
    rectangle = np.asarray(geometry.minimum_rotated_rectangle.exterior.coords)[:4]
    edges = np.roll(rectangle, -1, axis=0)-rectangle
    longest = edges[np.linalg.norm(edges, axis=1).argmax()]
    angle = (math.degrees(math.atan2(longest[1], longest[0]))+90) % 180-90
    return dict(height_clean=y1-y0, width_clean=x1-x0, area_clean=geometry.area,
                height_lr2=h_lr2, height_bin=HEIGHT_NAMES[sum(h_lr2 >= cut for cut in HEIGHT_CUTS)],
                aspect_aabb=(x1-x0)/max(y1-y0, 1e-12), orientation_long_edge_deg=angle,
                bounds=[x0, y0, x1, y1])


def diagnostic_label(tp, evidence, kept, candidate, cutoff, match_iou):
    if tp:
        return 'TP'
    if evidence['kernel_pixels'] < 4:
        return 'invalid_kernel'
    if evidence['coverage'] < .30:
        return 'weak_response'
    if kept and kept[0] >= match_iou:
        return 'assignment_ambiguous'
    if candidate and candidate[0] >= match_iou and candidate[1] < cutoff:
        return 'score_filtering'
    if kept and kept[0] > 0:
        return 'geometry'
    if candidate and candidate[0] > 0:
        return 'score_and_geometry'
    return 'no_candidate'


def instance_evidence(prediction, candidate, row, policy, kernel, score_cutoff):
    official, assigned, geometries, fp_indices = trace_matching(prediction, row, policy)
    kept_index = overlap_index(geometries, prediction['scores'])
    candidate_shapes, _ = filtered_geometry(candidate, row, policy)
    candidate_index = overlap_index(candidate_shapes, candidate['scores'])
    instances = {}
    for entry in row['gt_clean']['valid']:
        ann = str(entry['ann_id'])
        gt = shape(entry['geometry'])
        retained = overlaps(kept_index, gt)
        available = overlaps(candidate_index, gt)
        best, before = retained[0] if retained else None, available[0] if available else None
        match = assigned.get(ann)
        item = dict(tp=match is not None, assigned=match,
                    best_kept_iou=best[0] if best else 0., best_kept_score=best[1] if best else None,
                    candidate_iou=before[0] if before else 0., candidate_score=before[1] if before else None,
                    candidate_score_margin=before[1]-score_cutoff if before else None,
                    matching_neighbors=sum(iou >= policy['match_iou'] for iou, _, _ in retained))
        item['diagnosis'] = diagnostic_label(item['tp'], kernel[ann], best, before, score_cutoff, policy['match_iou'])
        instances[ann] = item
    return dict(counts=official['counts'], instances=instances, fp_indices=fp_indices, prediction=prediction)


def transition(before, after):
    return ('persistent_tp' if after else 'regressed') if before else ('rescued' if after else 'persistent_fn')


def fp_transitions(older, newer, row, policy):
    """Geometric correspondence of FP outputs, separate from detection matching."""
    old_geometries, _ = filtered_geometry(older['prediction'], row, policy)
    new_geometries, _ = filtered_geometry(newer['prediction'], row, policy)
    old_ids, new_ids = older['fp_indices'], newer['fp_indices']
    adjacency = []
    for old in old_ids:
        candidates = []
        for j, new in enumerate(new_ids):
            if old not in old_geometries or new not in new_geometries:
                continue
            g, p = old_geometries[old], new_geometries[new]
            inter = g.intersection(p).area
            iou = inter/(g.area+p.area-inter) if inter else 0.
            if iou >= .5:
                candidates.append((j, iou))
        adjacency.append([j for j, _ in sorted(candidates, key=lambda x: (-x[1], x[0]))])
    matches = maximum_matching(adjacency, len(new_ids))
    used_old, used_new = {i for i, _ in matches}, {j for _, j in matches}
    records = [dict(kind='persistent_fp', old_index=old_ids[i], new_index=new_ids[j]) for i, j in matches]
    records += [dict(kind='older_only_fp', old_index=p, new_index=None) for i, p in enumerate(old_ids) if i not in used_old]
    records += [dict(kind='newer_only_fp', old_index=None, new_index=p) for i, p in enumerate(new_ids) if i not in used_new]
    require(len(new_ids)-len(old_ids) == sum(r['kind']=='newer_only_fp' for r in records)-
            sum(r['kind']=='older_only_fp' for r in records), 'FP transition arithmetic failed')
    return records


def assert_transition_counts(counts, older_tp, newer_tp, gt_count):
    require(sum(counts.values()) == gt_count and
            counts.get('persistent_tp', 0)+counts.get('regressed', 0) == older_tp and
            counts.get('persistent_tp', 0)+counts.get('rescued', 0) == newer_tp,
            'Per-instance transitions do not reproduce TP/GT counts')
