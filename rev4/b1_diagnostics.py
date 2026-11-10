"""Paired B1-cal failure analysis on remaining validation."""
from collections import Counter
import copy
from pathlib import Path

from lowres.common import read_json, save_json, sha256
from rev4 import require
from rev4.b1_training import environment_identity
from rev4.calibration import CONDITIONS, validation_cohort
from rev4.stage_a import detector_for, forward


THRESHOLDS = (.30, .40)
HEIGHTS = (4, 6, 8, 16)
HEIGHT_BINS = ('<4', '4–<6', '6–<8', '8–<16', '≥16')


def reason(evidence, kept_ious, candidate_ious, iou_cutoff):
    if evidence['kernel_pixels'] < 4:
        return 'invalid_kernel'
    if evidence['coverage'] < .30:
        return 'weak_response'
    if any(iou >= iou_cutoff for iou in kept_ious):
        return 'ambiguous_matching'
    # The candidate pass uses the same mask, changing only the score cutoff.
    if any(iou >= iou_cutoff for iou in candidate_ious):
        return 'score_filtering'
    if kept_ious:
        return 'geometry'
    if candidate_ious:
        return 'ambiguous_score_and_geometry'
    return 'contour_postprocess'


def infer_pair(detector, processor, tasks, row, protocol, reference):
    from lowres.evaluate_shared import evaluate_image

    views = {}
    for condition in CONDITIONS:
        task = tasks[condition]
        _, probability, sample, factor, restore, prediction, _ = forward(detector, processor, task, row)
        scored = evaluate_image(row['gt_clean'], row['roi_clean'], prediction, protocol)
        require(scored['counts'] == reference[condition],
                f'B1-cal count mismatch against job 12396 for {task["sample_id"]}')
        views[condition] = dict(probability=probability, sample=sample, factor=factor, restore=restore,
                                prediction=prediction, matched=set(map(str, scored['matched_ann_ids'])))
    return views


def evidence_for(view, task, row, shrink):
    from mmocr.utils import offset_polygon
    from lowres.inference import kernel_evidence

    probability = view['probability'][0].detach().cpu().numpy()
    shape = tuple(view['sample'].metainfo['img_shape'][:2])
    evidence, valid, _ = kernel_evidence(probability, shape, view['factor'], task, row, task['size'],
                                         shrink, offset_polygon)
    return evidence, probability, valid


def analyze_source(candidate, views, tasks, row, protocol, shrink):
    from shapely.geometry import shape
    from lowres.inference import unpack
    from lowres.report import candidate_index, overlaps

    clean, lr2 = (views[c]['matched'] for c in CONDITIONS)
    lost = clean - lr2
    transitions = dict(clean_tp_lr2_tp=len(clean & lr2), clean_tp_lr2_fn=len(lost),
                       clean_fn_lr2_tp=len(lr2-clean), clean_fn_lr2_fn=len(row['gt_clean']['valid'])-len(clean | lr2))
    record = dict(source_image_id=row['source_image_id'], transitions=transitions, misses=[])
    if not lost:
        return record
    entries = [entry for entry in row['gt_clean']['valid'] if str(entry['ann_id']) in lost]
    require(len(entries) == transitions['clean_tp_lr2_fn'], 'Clean TP transition refers to unknown GT')
    residual = dict(row, gt_clean=dict(row['gt_clean'], valid=entries))
    clean_evidence, _, _ = evidence_for(views['clean'], tasks['clean'], residual, shrink)
    lr_evidence, _, _ = evidence_for(views['lr2'], tasks['lr2'], residual, shrink)
    view = views['lr2']
    unfiltered = unpack(candidate(view['probability'], [copy.deepcopy(view['sample'])])[0], view['restore'])
    kept = candidate_index(view['prediction'], row, protocol)
    candidates = candidate_index(unfiltered, row, protocol)
    for entry in entries:
        ann_id = str(entry['ann_id'])
        gt = shape(entry['geometry'])
        item = lr_evidence[ann_id]
        primary = reason(item, [iou for iou, _ in overlaps(kept, gt)],
                         [iou for iou, _ in overlaps(candidates, gt)], protocol['match_iou'])
        sensitivity = ('weak_response' if item['kernel_pixels'] >= 4 and item['coverage'] < .50 else primary)
        record['misses'].append(dict(source_image_id=row['source_image_id'], ann_id=ann_id,
                                     native_height=item['native_height'], network_input_height=item['network_input_height'],
                                     kernel_pixels=item['kernel_pixels'], coverage_clean=clean_evidence[ann_id]['coverage'],
                                     coverage_lr2=item['coverage'], primary=primary, sensitivity=sensitivity))
    return record


def summarize(records):
    transitions = Counter()
    for record in records:
        transitions.update(record['transitions'])
    misses = [item for record in records for item in record['misses']]
    require(len(misses) == transitions['clean_tp_lr2_fn'], 'Misses and Clean TP transitions disagree')
    by_height = {name: Counter() for name in HEIGHT_BINS}
    for miss in misses:
        bucket = HEIGHT_BINS[sum(miss['native_height'] >= bound for bound in HEIGHTS)]
        by_height[bucket][miss['primary']] += 1
    eligible = [miss for miss in misses if miss['native_height'] >= 6 and miss['kernel_pixels'] >= 4]
    weak = [miss for miss in eligible if miss['primary'] == 'weak_response']
    return dict(transitions=dict(transitions), primary_counts=dict(Counter(x['primary'] for x in misses)),
                sensitivity_counts=dict(Counter(x['sensitivity'] for x in misses)),
                by_native_height={k: dict(v) for k, v in by_height.items()},
                valid_kernel_at_least_6px=len(eligible), weak_at_least_6px=len(weak),
                weak_at_least_6px_clean_coverage_ge_050=sum(x['coverage_clean'] is not None and
                                                           x['coverage_clean'] >= .50 for x in weak))


def representative_cases(records):
    misses = [item for record in records for item in record['misses']]
    selected = []
    groups = (
        ('weak_response', 6, 8, 7),
        ('weak_response', 8, 16, 12),
        ('weak_response', 16, float('inf'), 24),
        ('score_filtering', 0, float('inf'), 12),
        ('geometry', 0, float('inf'), 12),
        ('contour_postprocess', 0, float('inf'), 12),
    )
    for category, lo, hi, midpoint in groups:
        choices = (x for x in misses if x['primary'] == category and lo <= x['native_height'] < hi)
        best = min(choices, key=lambda x: (abs(x['native_height']-midpoint), x['source_image_id'], x['ann_id']),
                   default=None)
        if best is not None:
            selected.append(best)
    return selected


def markdown(summary):
    transition = summary['transitions']
    lines = ['# B1-cal LR2 residual diagnostics', '',
             f'{summary["sources"]:,} paired remaining-validation sources. '
             'Mask 0.30, score 0.40. Every task reproduces job 12396 counts.', '',
             f'Clean TP → LR2 FN **{transition["clean_tp_lr2_fn"]:,}**. '
             f'Misses shared by Clean and LR2 **{transition["clean_fn_lr2_fn"]:,}**. '
             f'Clean FN → LR2 TP **{transition["clean_fn_lr2_tp"]:,}**.', '',
             '| Reason among Clean TP → LR2 FN | c0=0.30 | c0=0.50 |', '|---|---:|---:|']
    for name in sorted(summary['primary_counts'].keys() | summary['sensitivity_counts'].keys()):
        lines.append(f'| {name} | {summary["primary_counts"].get(name, 0):,} | '
                     f'{summary["sensitivity_counts"].get(name, 0):,} |')
    lines += ['', '| Native LR2 height | Misses | Weak response at c0=0.30 |', '|---|---:|---:|']
    for name in HEIGHT_BINS:
        counts = summary['by_native_height'][name]
        lines.append(f'| {name} px | {sum(counts.values()):,} | {counts.get("weak_response", 0):,} |')
    lines += ['', f'At height ≥6 px, **{summary["weak_at_least_6px"]:,}** of '
              f'**{summary["valid_kernel_at_least_6px"]:,}** valid-kernel misses have weak response.', '',
              f'Of those weak cases, **{summary["weak_at_least_6px_clean_coverage_ge_050"]:,}** have '
              'B1-cal Clean coverage ≥0.50. Review the selected crops before deciding on PRC. '
              'This is a descriptive diagnosis on the validation used for calibration.', '']
    return '\n'.join(lines)


def run(a):
    prepared = a.calibration_prepared
    rows, tasks = validation_cohort(a, prepared)
    require(environment_identity(a, prepared) == a.b1_identity, 'Require the audited B1 runtime')
    reference_counts = read_json(a.calibration_counts)
    identity = dict(calibration_sha256=a.calibration_sha256, calibration_counts_sha256=a.calibration_counts_sha256,
                    thresholds=list(THRESHOLDS), worker_sha256=sha256(Path(__file__)),
                    review_sha256=sha256(Path(__file__).with_name('b1_diagnostic_review.py')))
    require(reference_counts['identity'] == a.calibration_summary['identity'] and
            len(reference_counts['counts']) == len(tasks), 'Calibration counts do not cover the locked cohort')
    grid_index = reference_counts['identity']['grid'].index(list(THRESHOLDS))
    paired_tasks = {source: {} for source in rows}
    paired_counts = {source: {} for source in rows}
    for task, counts in zip(tasks, reference_counts['counts']):
        source, condition = task['source_image_id'], task['condition']
        paired_tasks[source][condition] = task
        paired_counts[source][condition] = counts[grid_index]
    sources = sorted(rows)
    protocol = read_json(a.evaluation_protocol)
    cache = a.diagnostics_run/'paired_records.json'
    progress = read_json(cache) if cache.is_file() else dict(identity=identity, records=[])
    records = progress['records']
    require(progress['identity'] == identity and [x['source_image_id'] for x in records] == sources[:len(records)],
            'Cached diagnostics differ from the locked cohort or code')
    options = copy.copy(a)
    options.checkpoint = a.b1_checkpoint
    detector, processor, candidate, shrink = detector_for(options)
    processor.mask_thr, processor.min_text_score = THRESHOLDS

    def infer_source(source):
        return infer_pair(detector, processor, paired_tasks[source], rows[source], protocol, paired_counts[source])

    for index, source in enumerate(sources[len(records):], len(records)+1):
        records.append(analyze_source(candidate, infer_source(source), paired_tasks[source], rows[source],
                                      protocol, shrink))
        if index % 50 == 0 or index == len(sources):
            save_json(cache, progress)
            print(f'B1-cal diagnostic {index:,}/{len(sources):,} sources', flush=True)

    from rev4.b1_diagnostic_review import save_sheet

    result = a.diagnostics_run/'results'
    result.mkdir(parents=True, exist_ok=True)
    selected = representative_cases(records)
    findings = summarize(records)
    transitions = findings['transitions']
    scores = a.calibration_summary['models']['B1']['calibrated']['scores']
    require(transitions['clean_tp_lr2_tp']+transitions['clean_tp_lr2_fn'] == scores['clean']['tp'] and
            transitions['clean_tp_lr2_tp']+transitions['clean_fn_lr2_tp'] == scores['lr2']['tp'] and
            transitions['clean_fn_lr2_fn']+transitions['clean_tp_lr2_fn'] == scores['lr2']['fn'],
            'Paired transitions disagree with the calibration summary')
    if selected:
        save_sheet(result/'b1_cal_pmap_review.png', selected, rows, paired_tasks, infer_source, shrink)
    summary = dict(identity=identity, sources=len(sources), calibration_job='12396',
                   calibrated_scores=scores, examples=selected, **findings)
    (result/'b1_cal_diagnostics.md').write_text(markdown(summary), encoding='utf-8')
    save_json(result/'b1_cal_diagnostics.json', summary)
    print('B1-cal diagnostics complete:', result/'b1_cal_diagnostics.md', flush=True)
