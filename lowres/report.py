"""Validation-only paired metrics and the predeclared FROZEN v2.0 decision gates."""
from collections import Counter

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .evaluate_shared import rates


def candidate_index(prediction, row, protocol):
    roi = shape(row['roi_clean'])
    ignored = unary_union([shape(item['geometry']) for item in row['gt_clean']['ignored']])
    shapes, scores = [], []
    for coords, score in zip(prediction['polygons'], prediction['scores']):
        geometry = Polygon(np.asarray(coords).reshape(-1, 2))
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty or geometry.area <= 0:
            continue
        clipped = geometry.intersection(roi)
        if clipped.is_empty or clipped.area < protocol['min_prediction_roi_fraction']*geometry.area:
            continue
        if ignored.intersection(clipped).area > protocol['ignore_prediction_overlap']*clipped.area:
            continue
        shapes.append(clipped)
        scores.append(score)
    return shapes, scores, STRtree(shapes)


def overlaps(index, gt):
    shapes, scores, tree = index
    found = []
    for i in tree.query(gt, predicate='intersects'):
        geometry = shapes[i]
        intersection = geometry.intersection(gt).area
        if intersection:
            found.append((intersection/(gt.area+geometry.area-intersection), scores[i]))
    return found


def classify(evidence, gt, kept, candidates, response_cutoff, iou_cutoff):
    if evidence['kernel_pixels'] < 4:
        return 'invalid_kernel'
    if evidence['coverage'] < response_cutoff:
        return 'weak_response'
    retained = overlaps(kept, gt)
    if any(iou >= iou_cutoff for iou, _ in retained):
        return 'ambiguous_matching'
    eligible = overlaps(candidates, gt)
    if any(iou >= iou_cutoff and score < .30 for iou, score in eligible):
        return 'score_filtering'
    if retained:
        return 'geometry'
    if eligible:
        return 'ambiguous_score_and_geometry'
    return 'contour_postprocess'


def metrics(rows):
    tp, gt, predicted, coverage = np.asarray([row['metrics']['counts'] for row in rows]).sum(axis=0)
    return dict(tp=int(tp), fp=int(predicted-tp), fn=int(gt-tp), **rates([tp, gt, predicted, coverage]))


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def bootstrap(results, sources, repetitions=1000):
    rng = np.random.default_rng(20261107)
    indices = rng.integers(0, len(sources), size=(repetitions, len(sources)))
    means = {}
    for condition, rows in results.items():
        per_image = np.asarray([rows[s]['metrics']['counts'][:3] for s in sources])
        tp, gt, predicted = per_image[indices].sum(axis=1).T
        means[condition] = 2*tp/np.maximum(gt+predicted, 1)
    return means


def diagnostic(results, clean, gt_rows, sources, protocol):
    surviving, total_clean_tp = 0, 0
    lost = []
    bins = {name: dict(gt=0, matched=0) for name in ('lt4', '4_to_lt8', '8_to_lt16', 'ge16')}
    for source in sources:
        record, reference = results[source], clean[source]
        matched = set(map(str, record['metrics']['matched_ann_ids']))
        clean_matched = set(map(str, reference['metrics']['matched_ann_ids']))
        total_clean_tp += len(clean_matched)
        surviving += len(matched & clean_matched)
        for annotation, item in record['gt'].items():
            height = item['native_height']
            bin_name = list(bins)[sum(height >= bound for bound in (4, 8, 16))]
            bins[bin_name]['gt'] += 1
            bins[bin_name]['matched'] += annotation in matched
        missing = clean_matched - matched
        if not missing:
            continue
        row = gt_rows[source]
        kept = candidate_index(record['prediction'], row, protocol)
        candidates = candidate_index(record['candidates_before_score'], row, protocol)
        for entry in row['gt_clean']['valid']:
            annotation = str(entry['ann_id'])
            if annotation not in missing:
                continue
            item = record['gt'][annotation]
            clean_coverage = reference['gt'][annotation]['coverage']
            coverage = item['coverage']
            gt = shape(entry['geometry'])
            lost.append(dict(source_image_id=source, ann_id=annotation, native_height=item['native_height'],
                             network_input_height=item['network_input_height'], kernel_pixels=item['kernel_pixels'],
                             coverage_clean=clean_coverage, coverage_lr=coverage,
                             delta_coverage=coverage-clean_coverage if None not in (coverage, clean_coverage) else None,
                             primary=classify(item, gt, kept, candidates, .30, protocol['match_iou']),
                             sensitivity=classify(item, gt, kept, candidates, .50, protocol['match_iou'])))
    for row in bins.values():
        row['recall'] = row['matched']/row['gt'] if row['gt'] else None
    return dict(clean_tp=total_clean_tp, clean_tp_survival=surviving/total_clean_tp if total_clean_tp else None,
                clean_tp_to_fn=len(lost), native_height_bins=bins, failures=lost)


def summarize(results, gt_rows, protocol):
    sources = sorted(gt_rows)
    means = bootstrap(results, sources)
    clean = results['clean']
    summaries = {}
    for condition, rows in results.items():
        result = metrics(rows.values())
        result['latency_median_ms_decode_forward_native_postprocess'] = float(
            np.median([row['latency_ms'] for row in rows.values()]))
        if condition != 'clean':
            result.update(diagnostic(rows, clean, gt_rows, sources, protocol))
            result['primary_failure_counts'] = dict(Counter(item['primary'] for item in result['failures']))
            result['sensitivity_failure_counts'] = dict(Counter(item['sensitivity'] for item in result['failures']))
        summaries[condition] = result
    baseline, lr4, up = (summaries[c] for c in ('clean', 'lr4', 'b_up_lr4'))
    for condition, result in summaries.items():
        result['hmean_drop_pp'] = 100*(baseline['hmean']-result['hmean'])
        result['recall_drop_pp'] = 100*(baseline['recall']-result['recall'])
        result['hmean_drop_95ci_pp'] = interval(100*(means['clean']-means[condition]))
    ci = lr4['hmean_drop_95ci_pp']
    recovery = None if ci[0] <= 0 <= ci[1] else (up['hmean']-lr4['hmean'])/(baseline['hmean']-lr4['hmean'])
    losses = lr4['failures']
    heights = {str(h): sum(item['native_height'] >= h for item in losses)/len(losses) if losses else None
               for h in (4, 6, 8)}
    lost_fraction = len(losses)/lr4['clean_tp'] if lr4['clean_tp'] else None
    valid = [item for item in losses if item['kernel_pixels'] >= 4]
    weak = {key: sum(item[key] == 'weak_response' for item in valid)/len(valid) if valid else None
            for key in ('primary', 'sensitivity')}
    clean_coverage = {str(cutoff): sum(item['coverage_clean'] is not None and item['coverage_clean'] >= cutoff
                                      for item in losses)/len(losses) if losses else None for cutoff in (.30, .50)}
    return dict(
        source_count=len(sources), bootstrap_repetitions=1000, bootstrap_seed=20261107, conditions=summaries,
        lr4_b_up_recovery_ratio=recovery, lr4_clean_tp_to_fn_fraction=lost_fraction,
        lr4_lost_fraction_at_min_native_height=heights, lr4_weak_response_fraction_valid_kernel=weak,
        lr4_lost_fraction_clean_coverage_at_cutoff=clean_coverage,
        gates=dict(G1=bool(lr4['hmean_drop_pp'] >= 5 and ci[0] > 0),
                   G2=bool(lost_fraction is not None and lost_fraction >= .15),
                   G3=bool(heights['6'] is not None and heights['6'] >= .30),
                   G4='manual review of B_up, maps, word sizes and examples'),
        intervention='pending pilot review')


def markdown(summary):
    lines = ['# FULL_R1 Low Resolution pilot',
             '', f'Validation sources: {summary["source_count"]}. B0 FP32. Paired bootstrap by source.',
             '', '| Condition | Precision | Recall | Hmean | TP | FP | FN | Clean TP survival |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in summary['conditions'].items():
        survival = row.get('clean_tp_survival')
        label = 'N/A' if survival is None else f'{survival:.2%}'
        lines.append(f'| {name} | {row["precision"]:.2%} | {row["recall"]:.2%} | {row["hmean"]:.2%} | '
                     f'{row["tp"]} | {row["fp"]} | {row["fn"]} | {label} |')
    lr4 = summary['conditions']['lr4']
    lo, hi = lr4['hmean_drop_95ci_pp']
    recovery = summary['lr4_b_up_recovery_ratio']
    recovery_label = 'N/A' if recovery is None else f'{recovery:.3f}'
    lines += ['', f'LR4 Hmean drop {lr4["hmean_drop_pp"]:.2f} pp, 95% CI [{lo:.2f}, {hi:.2f}].',
              f'B_up recovery ratio {recovery_label}.', '']
    for gate, status in summary['gates'].items():
        label = ('PASS' if status else 'FAIL') if isinstance(status, bool) else status
        lines.append(f'- {gate} {label}')
    lines += ['', 'Review LR4 samples and P-maps before making an intervention decision.', '']
    return '\n'.join(lines)
