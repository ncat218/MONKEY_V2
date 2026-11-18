"""Descriptive paired-instance reports; no fitting, new thresholds or TEST data."""
from collections import Counter, defaultdict
import csv
import hashlib
import json

import numpy as np

from lowres.common import read_json, save_json
from lowres.evaluate_shared import bootstrap_paired, rates
from rev4 import require
from rev4.adaptation_plan import BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, HEIGHT_NAMES, MODEL_PAIRS
from rev4.paired_eda_core import assert_transition_counts, fp_transitions, transition

MODES = ('native', 'common', 'calibrated')
GROUPS = ('rescued', 'regressed', 'persistent_fn', 'persistent_tp')


def metric(counts):
    tp, gt, pred, _ = map(int, counts)
    return dict(**rates(counts), tp=tp, fp=pred-tp, fn=gt-tp, gt=gt)


def summarize(a, models, rows, tasks):
    from rev4.paired_eda import mode_result

    output = a.output/'results'/a.mode
    output.mkdir(parents=True, exist_ok=True)
    names = [m['name'] for m in models]
    pairs = [(old, new) for old, new in MODEL_PAIRS if old in names and new in names]
    sources = sorted(rows)
    require(len(sources) >= 2, 'Paired report needs at least two sources')
    source_index = {source: index for index, source in enumerate(sources)}
    matrices = {(name, mode, condition): np.zeros((len(sources), 4), dtype=np.int64)
                for name in names for mode in MODES for condition in ('clean', 'lr2')}
    transitions = defaultdict(Counter)
    false_positives = defaultdict(Counter)
    strata = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    diagnoses = defaultdict(Counter)
    distributions = defaultdict(list)
    examples = defaultdict(list)
    policy = read_json(a.evaluation_protocol)
    fields = ['older', 'newer', 'mode', 'condition', 'source_image_id', 'ann_id', 'group',
              'height_bin', 'height_clean', 'height_lr2', 'width_clean', 'area_clean', 'aspect_aabb',
              'orientation_long_edge_deg', 'old_tp', 'new_tp', 'old_assigned_iou', 'new_assigned_iou',
              'old_assigned_score', 'new_assigned_score']
    evidence_fields = ('best_kept_iou', 'best_kept_score', 'candidate_iou', 'candidate_score',
                       'candidate_score_margin', 'matching_neighbors', 'diagnosis')
    fields += [f'{side}_{key}' for side in ('old', 'new') for key in evidence_fields]
    fields += [f'{side}_{key}' for side in ('old', 'new')
               for key in ('kernel_pixels', 'coverage', 'p_mean', 'p_q10', 'p_median', 'p_q90')]
    with ((output/'paired_transitions.csv').open('w', newline='', encoding='utf-8') as stream,
          (output/'fp_analysis.csv').open('w', newline='', encoding='utf-8') as fp_stream,
          (output/'input_shapes.csv').open('w', newline='', encoding='utf-8') as shape_stream):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        fp_writer = csv.DictWriter(fp_stream, fieldnames=['older', 'newer', 'mode', 'condition',
                'source_image_id', 'kind', 'old_index', 'new_index', 'old_score', 'new_score',
                'old_polygon', 'new_polygon'])
        fp_writer.writeheader()
        shape_writer = csv.DictWriter(shape_stream, fieldnames=['model', 'condition', 'source_image_id',
                'native_size', 'resized_hw', 'padded_input', 'pmap', 'backbone', 'neck'])
        shape_writer.writeheader()
        for number, task in enumerate(tasks, 1):
            source, condition = task['source_image_id'], task['condition']
            records = {name: read_json(a.output/'models'/name/condition/f'{source}.json') for name in names}
            for name, record in records.items():
                require(record['sample_id'] == task['sample_id'], 'EDA report task order mismatch')
                shape_writer.writerow(dict(model=name, condition=condition, source_image_id=source,
                            **{key: json.dumps(value) for key, value in record['dimensions'].items()}))
                for mode in MODES:
                    result = mode_result(record, mode)
                    matrices[name, mode, condition][source_index[source]] = result['counts']
                    for ann, evidence in result['instances'].items():
                        bucket = record['gt_attributes'][ann]['height_bin']
                        strata[name, mode, condition, bucket] += [1, int(evidence['tp'])]
                        diagnoses[name, mode, condition][evidence['diagnosis']] += 1
            for older, newer in pairs:
                old, new = records[older], records[newer]
                require(old['gt_attributes'] == new['gt_attributes'], 'Models disagree on GT attributes')
                for mode in MODES:
                    before, after = mode_result(old, mode), mode_result(new, mode)
                    require(set(before['instances']) == set(after['instances']), 'Models have different GT IDs')
                    current = Counter()
                    for ann, previous in before['instances'].items():
                        following = after['instances'][ann]
                        group = transition(previous['tp'], following['tp'])
                        current[group] += 1
                        attributes = old['gt_attributes'][ann]
                        item = dict(older=older, newer=newer, mode=mode, condition=condition,
                                    source_image_id=source, ann_id=ann, group=group,
                                    **{key: value for key, value in attributes.items() if key != 'bounds'},
                                    old_tp=previous['tp'], new_tp=following['tp'])
                        for side, record, evidence in [('old', old, previous), ('new', new, following)]:
                            assigned = evidence['assigned'] or {}
                            item.update({f'{side}_assigned_iou': assigned.get('iou'),
                                         f'{side}_assigned_score': assigned.get('score')})
                            item.update({f'{side}_{key}': evidence[key] for key in evidence_fields})
                            kernel = record['kernel_evidence'][ann]
                            item.update({f'{side}_{key}': kernel[key] for key in ('kernel_pixels', 'coverage')})
                            item.update({f'{side}_p_{key}': value for key, value in
                                        record['polygon_probability_statistics'][ann].items()})
                            if not evidence['tp']:
                                distributions[older, newer, mode, condition, group, side, 'candidate_iou'].append(evidence['candidate_iou'])
                                distributions[older, newer, mode, condition, group, side, 'candidate_score_margin'].append(evidence['candidate_score_margin'])
                        writer.writerow(item)
                        if mode == 'common':
                            # Deterministic hash sampling within each group/height bin, never hand-picked wins.
                            key = (older, newer, condition, group, attributes['height_bin'])
                            rank = hashlib.sha256(f'{BOOTSTRAP_SEED}:{source}:{ann}'.encode()).hexdigest()
                            examples[key].append((rank, dict(older=older, newer=newer, condition=condition,
                                    source_image_id=source, ann_id=ann, group=group, height_bin=attributes['height_bin'])))
                            examples[key].sort(key=lambda x: x[0])
                            examples[key] = examples[key][:2]
                    assert_transition_counts(current, before['counts'][0], after['counts'][0], before['counts'][1])
                    transitions[older, newer, mode, condition].update(current)
                    fp_records = fp_transitions(before, after, rows[source], policy)
                    false_positives[older, newer, mode, condition].update(r['kind'] for r in fp_records)
                    for fp in fp_records:
                        item = dict(older=older, newer=newer, mode=mode, condition=condition,
                                    source_image_id=source, **fp)
                        for side, result in [('old', before), ('new', after)]:
                            index = fp[f'{side}_index']
                            item[f'{side}_score'] = result['prediction']['scores'][index] if index is not None else None
                            item[f'{side}_polygon'] = json.dumps(result['prediction']['polygons'][index]) if index is not None else ''
                        fp_writer.writerow(item)
            if number % 500 == 0:
                print(f'EDA reports: {number}/{len(tasks)} tasks', flush=True)
    scores = {name: {mode: {condition: metric(matrices[name, mode, condition].sum(axis=0))
                           for condition in ('clean', 'lr2')} for mode in MODES} for name in names}
    comparisons = []
    for older, newer in pairs:
        for mode in MODES:
            for condition in ('clean', 'lr2'):
                old_matrix, new_matrix = matrices[older, mode, condition], matrices[newer, mode, condition]
                totals = transitions[older, newer, mode, condition]
                assert_transition_counts(totals, int(old_matrix[:, 0].sum()), int(new_matrix[:, 0].sum()),
                                         int(old_matrix[:, 1].sum()))
                bootstrap = bootstrap_paired(np.concatenate([old_matrix, new_matrix], axis=1),
                                             BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED)
                comparisons.append(dict(older=older, newer=newer, mode=mode, condition=condition,
                        hmean_difference_pp=100*(scores[newer][mode][condition]['hmean']-scores[older][mode][condition]['hmean']),
                        paired_source_bootstrap=bootstrap, transitions=dict(totals),
                        fp_correspondence=dict(false_positives[older, newer, mode, condition])))
    write_metrics(output, scores, strata)
    selected = [case for key in sorted(examples) for _, case in examples[key]]
    save_json(output/'selected_examples.json', dict(selection='two lowest SHA256 ranks per model pair, condition, transition and fixed LR2 height bin', examples=selected))
    histogram_rows = []
    for key, values in sorted(distributions.items()):
        valid = [v for v in values if v is not None]
        edges = np.linspace(0, 1, 21) if key[-1] == 'candidate_iou' else np.linspace(-1, 1, 41)
        histogram_rows.append(dict(older=key[0], newer=key[1], mode=key[2], condition=key[3],
                        group=key[4], side=key[5], variable=key[6], missing=len(values)-len(valid),
                        bin_edges=edges.tolist(), counts=np.histogram(valid, edges)[0].tolist(),
                        quantiles=dict(zip(('p10', 'p50', 'p90'), np.quantile(valid, [.1, .5, .9]).tolist())) if valid else None))
    save_json(output/'failure_distributions.json', histogram_rows)
    summary = dict(sources=len(rows), models=models, scores=scores, comparisons=comparisons,
                   diagnoses=[dict(model=key[0], mode=key[1], condition=key[2], counts=dict(value))
                              for key, value in sorted(diagnoses.items())], no_test=True,
                   validation_role='development; already used for checkpoint/threshold and method selection',
                   height_definition='GT AABB height on the native LR2 grid, used to stratify both conditions',
                   examples=len(selected), all_historical_per_image_counts_reproduced=True,
                   uncertainty='paired source CI conditional on trained checkpoints; not across-seed uncertainty')
    save_json(output/'eda_summary.json', summary)
    report(output, scores, comparisons)
    return selected, output


def write_metrics(output, scores, strata):
    with (output/'model_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['model', 'mode', 'condition', 'precision', 'recall', 'hmean', 'word_coverage_recall', 'tp', 'fp', 'fn', 'gt']
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for name, modes in scores.items():
            for mode, conditions in modes.items():
                for condition, values in conditions.items():
                    writer.writerow(dict(model=name, mode=mode, condition=condition, **values))
    with (output/'stratified_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, ['model', 'mode', 'condition', 'height_bin', 'gt', 'tp', 'recall'])
        writer.writeheader()
        for (name, mode, condition, bucket), (gt, tp) in sorted(strata.items()):
            writer.writerow(dict(model=name, mode=mode, condition=condition, height_bin=bucket,
                                  gt=int(gt), tp=int(tp), recall=float(tp/gt) if gt else None))


def report(output, scores, comparisons):
    lines = ['# Paired B0 / B-Clean / B1 adaptation EDA', '',
             'Development data only. All per-image counts must reproduce the saved calibration counts.',
             'Native = 0.30/0.30; common = 0.30/0.40; calibrated = each model\'s selected original 15-pair grid.', '',
             '| Model | Mode | Clean Hmean | LR2 Hmean | Clean FP | LR2 FP |', '|---|---|---:|---:|---:|---:|']
    for name, modes in scores.items():
        for mode, values in modes.items():
            lines.append(f'| {name} | {mode} | {values["clean"]["hmean"]:.2%} | {values["lr2"]["hmean"]:.2%} | {values["clean"]["fp"]} | {values["lr2"]["fp"]} |')
    lines += ['', '| Comparison (new minus old) | Condition | Hmean difference pp [95% CI] | Rescued | Regressed | Persistent FN |',
              '|---|---|---:|---:|---:|---:|']
    for value in comparisons:
        if value['mode'] != 'common':
            continue
        lo, hi = value['paired_source_bootstrap']['delta_pp_95ci']['hmean']
        counts = value['transitions']
        lines.append(f'| {value["older"]} → {value["newer"]} | {value["condition"]} | {value["hmean_difference_pp"]:+.3f} [{lo:+.3f}, {hi:+.3f}] | {counts.get("rescued", 0)} | {counts.get("regressed", 0)} | {counts.get("persistent_fn", 0)} |')
    lines += ['', 'CI: 2,000 paired source bootstrap replicates, seed 20261107. Post-selection descriptive intervals, not training-seed uncertainty.',
              'FP correspondence matches FP polygons across models at IoU ≥ 0.5 for description; it does not change evaluation.',
              'P-map statistics and geometric categories are diagnostic proxies. Inspect selected image/P-map pairs before inferring causes.',
              'Two examples per height bin and transition are chosen deterministically, including regressions; this is not a random prevalence estimate.',
              'Raw prediction dimensions and actual backbone/FPNC shapes are in input_shapes.csv. EDA timing is not a latency benchmark.', '',
              'Artifacts: paired_transitions.csv, fp_analysis.csv, model_metrics.csv, stratified_metrics.csv,',
              'failure_distributions.json, selected_examples.json and examples/ (paired image/P-map crops).', '']
    (output/'eda_report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'mechanism_selection.md').write_text(
        '# Evidence required before choosing a new mechanism\n\n'
        '1. Compare B0 → B-Clean to isolate ordinary domain fine-tuning, then B-Clean → B1 for the incremental explicit LR2 mixture.\n'
        '2. Inspect rescued/regressed/persistent-FN pairs by fixed LR2 word height, orientation and aspect ratio.\n'
        '3. Distinguish weak kernel response, candidate confidence and candidate IoU; missing candidate confidence is missing, not zero.\n'
        '4. If candidate overlap is already adequate but confidence is low, examine score/postprocessing controls before architecture changes.\n'
        '5. If strong-response words still have inadequate candidate IoU, examine geometry and scale alignment before an adapter.\n'
        '6. If B-Clean nearly matches B1, describe transfer from Clean before claiming resolution-specific adaptation.\n\n'
        'These are conditional research decisions, not demonstrated causes or automatic authorization for further training.\n', encoding='utf-8')
