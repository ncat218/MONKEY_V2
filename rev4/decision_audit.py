"""Exploratory quad/threshold audit and paired error review on remaining validation."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from lowres.common import identity as fingerprint, read_json, save_json, sha256
from rev4 import require
from rev4.adaptation_jobs import lock_record
from rev4.calibration import validation_cohort
from rev4.decision_core import (CandidateMatcher, capture_candidates, choose_setting,
                                detection_rates, paired_hmean_ci, setting_key, summarize_counts)
from rev4.decision_plan import (BOOTSTRAP_SEED, BOOTSTRAP_REPETITIONS, COMMON_INDEX,
                                GEOMETRIES, INSTALLED_PP_SHA256, SCORES, STAGE)
from rev4.decision_review import combine_panels, prepare_review, render_panels
from rev4.paired_eda import mode_result
from rev4.stage_a import detector_for, forward


def installed_postprocessor():
    spec = importlib.util.find_spec('mmocr.models.textdet.postprocessors.db_postprocessor')
    require(spec is not None and spec.origin, 'Cannot locate installed MMOCR DBPostprocessor')
    path = Path(spec.origin)
    require(sha256(path) == INSTALLED_PP_SHA256,
            f'Installed DBPostprocessor hash changed: {sha256(path)}')
    return path


def model_contexts(a):
    eda = read_json(a.eda_root/'results/B0_BClean_B1/complete.json')
    require(eda.get('no_test') is True and eda['cohort']['sources'] == 2262 and
            eda['cohort']['validation_sha256'] == sha256(a.calibration_prepared/'validation_remaining.jsonl'),
            'Historical EDA is not the locked 2,262-source remaining-validation cohort')
    require(eda['cohort']['policy_sha256'] == sha256(a.evaluation_protocol),
            'Historical EDA evaluator policy differs')
    contexts = []
    for name in ('B0', 'B-Clean', 'B1'):
        identity_path = a.eda_root/'models'/name/'identity.json'
        identity = read_json(identity_path)
        model = identity['model']
        require(model['name'] == name and tuple(model['calibrated_pair']) == (.30, .40),
                f'{name} EDA identity or calibrated threshold changed')
        checkpoint = Path(model['checkpoint'])
        require(checkpoint.is_file() and sha256(checkpoint) == model['checkpoint_sha256'],
                f'{name} checkpoint hash mismatch')
        contexts.append(dict(name=name, checkpoint=checkpoint, checkpoint_sha256=model['checkpoint_sha256'],
                             eda_model=model, identity=identity['model'], identity_sha256=fingerprint(identity)))
    require([m['name'] for m in contexts] == [m['name'] for m in eda['models']],
            'EDA model order changed')
    return contexts, eda


def cached_baseline(a, model, task):
    path = a.eda_root/'models'/model['name']/task['condition']/f'{task["source_image_id"]}.json'
    record = read_json(path)
    require(record['identity_sha256'] == model['identity_sha256'] and
            record['sample_id'] == task['sample_id'] and record['image_sha256'] == task['image_sha256'],
            f'Historical EDA cache mismatch: {model["name"]}/{task["sample_id"]}')
    result = mode_result(record, 'common')
    return record, result


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def report_text(a, models, selected_rows, comparisons, review_summary, pp_path):
    lines = ['# PR, quad postprocessing and paired error diagnosis', '',
        'Exploratory analysis on the same 2,262 remaining-validation sources previously used for model selection, calibration and EDA. '
        'This is not a fresh holdout. S1/S2 were already viewed in pilot work. TEST was not read. No training was run.', '',
        f'Installed MMOCR DBPostprocessor: `{pp_path}` (SHA-256 `{INSTALLED_PP_SHA256}`). '
        'The audited inference path is `quad`; the original protocol/evaluator were not changed.', '',
        '## Existing calibrated baseline and exploratory selection', '',
        '| Model | Setting selected within this exploratory grid | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP | LR2 recall <4 px |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ('B0', 'B-Clean', 'B1'):
        row = selected_rows[name]
        p = row['setting']
        lines.append(f'| {name} | {setting_key(p)}; mask {p["mask_thr"]:.2f}, unclip {p["unclip_ratio"]:.2f}, '
                     f'min width {p["min_text_width"]}, epsilon {p["epsilon_ratio"]:.3f}, score {p["min_text_score"]:.2f} | '
                     f'{100*row["clean"]["hmean"]:.2f}% | {100*row["lr2"]["hmean"]:.2f}% | '
                     f'{100*row["lr2"]["precision"]:.2f}% | {100*row["lr2"]["recall"]:.2f}% | '
                     f'{row["lr2"]["fp"]:,} | {100*row["lr2"]["tiny_recall"]:.2f}% |')
    lines += ['', 'Selection rule: maximize the unweighted mean of Clean and LR2 micro-Hmean over the same grid for each model; '
              'the original base `.30/.40` setting wins exact ties. This is exploratory selection on already-used validation, '
              'not an independent performance claim.', '', '## Change from original calibrated `.30/.40`', '',
              '| Model | Condition | Δ Hmean (pp) | Paired source-bootstrap 95% CI |', '|---|---|---:|---|']
    for name in ('B0', 'B-Clean', 'B1'):
        for condition in ('clean', 'lr2'):
            item = comparisons[name][condition]
            interval = item['ci95_pp']
            lines.append(f'| {name} | {condition} | {item["delta_pp"]:+.3f} | '
                         f'[{interval[0]:+.3f}, {interval[1]:+.3f}] |')
    lines += ['', '## Candidate and GT diagnosis', '',
              'The taxonomy CSV contains automatic geometric proxies, not verified causes. '
              'The human review sheet is pending and must be reviewed before treating categories as visual findings.', '',
              f'- Sampled review cards: {review_summary["sampled_cases"]:,}; status: PENDING.',
              '- FP proxy categories and GT transition counts are in `diagnosis_summary.json`.',
              '- Per-prediction/GT rows are in `fp_taxonomy_proxies.csv` and `gt_diagnosis.csv`.',
              '- Human labels and reviewer notes belong in `human_review.csv`; preserve unreviewed labels as blank.', '',
              '## Artifacts', '',
              '- `pr_curves.csv`: base quad PR/Hmean and FP versus <4 px recall over score thresholds.',
              '- `threshold_frontier.csv`: all exploratory geometry and score combinations.',
              '- `quad_filter_audit.csv`: contour cap, approximation, score, unclip and min-width flow counters.',
              '- `review/`: stratified image/P-map pairs for human inspection.',
              '- `run_identity.json`: source, checkpoint, data and installed postprocessor hashes.', '',
              'No training or TEST evaluation was performed. Do not replace the frozen `.30/.40` reports with these exploratory results.', '']
    return '\n'.join(lines)


def run(a):
    import torch
    from lowres.common import read_json as read

    pp_path = installed_postprocessor()
    rows, tasks = validation_cohort(a, a.calibration_prepared)
    require(len(rows) == 2262 and len(tasks) == 4524, 'Expected only remaining-validation Clean/LR2 tasks')
    models, eda = model_contexts(a)
    policy = read(a.evaluation_protocol)
    source_ids = sorted(rows)
    source_index = {source: i for i, source in enumerate(source_ids)}
    task_by_pair = {(task['source_image_id'], task['condition']): task for task in tasks}
    require(len(task_by_pair) == len(tasks) and all((s, c) in task_by_pair for s in source_ids for c in ('clean', 'lr2')),
            'Remaining-validation task pairs are incomplete')
    identity = dict(stage=STAGE, original_protocol_sha256=sha256(a.evaluation_protocol),
        validation_sha256=sha256(a.calibration_prepared/'validation_remaining.jsonl'),
        task_manifest_sha256=sha256(a.calibration_prepared/'validation_tasks.jsonl'),
        eda_complete_sha256=sha256(a.eda_root/'results/B0_BClean_B1/complete.json'),
        installed_db_postprocessor_path=str(pp_path), installed_db_postprocessor_sha256=INSTALLED_PP_SHA256,
        grid=[dict(name=g[0], mask_thr=g[1], unclip_ratio=g[2], min_text_width=g[3], epsilon_ratio=g[4]) for g in GEOMETRIES],
        scores=list(SCORES), models=[dict(name=m['name'], checkpoint_sha256=m['checkpoint_sha256'],
                                           eda_identity_sha256=m['identity_sha256']) for m in models],
        source_code=a.decision_source_identity, seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPETITIONS,
        no_test=True, training=False, human_review='PENDING')
    identity = lock_record(a.output/'run_identity.json', identity)
    result_root = a.output/'results'
    result_root.mkdir(parents=True, exist_ok=True)
    matrices = {}
    audit_totals = {(m['name'], g[0], c): dict(contours=0, examined=0, candidate_limit_skipped=0,
                    approximation_rejected=0, scored=0, unclip_empty=0, width_rejected=0, accepted=0)
                    for m in models for g in GEOMETRIES for c in ('clean', 'lr2')}
    settings = []
    for geometry in GEOMETRIES:
        for score in SCORES:
            settings.append(dict(geometry=geometry[0], mask_thr=geometry[1], unclip_ratio=geometry[2],
                min_text_width=geometry[3], epsilon_ratio=geometry[4], min_text_score=score))
    for model in models:
        print(f'Loading {model["name"]}; checkpoint {model["checkpoint_sha256"]}', flush=True)
        options = copy.copy(a)
        options.checkpoint = model['checkpoint']
        detector, native, _, _ = detector_for(options)
        require(native.text_repr_type == 'quad' and (native.mask_thr, native.min_text_score,
                native.unclip_ratio, native.min_text_width, native.epsilon_ratio) == (.30, .30, 1.5, 5, .01),
                'Native DBPostprocessor differs from the frozen quad configuration')
        matrix = np.zeros((len(source_ids), 2, len(settings), 5), dtype=np.int64)
        for number, task in enumerate(tasks, 1):
            row = rows[task['source_image_id']]
            cached, baseline = cached_baseline(a, model, task)
            _, probability, sample, _, restore, _, _ = forward(detector, native, task, row)
            condition_index = 0 if task['condition'] == 'clean' else 1
            source_i = source_index[task['source_image_id']]
            for geometry_i, geometry in enumerate(GEOMETRIES):
                prediction, raw_scores, audit = capture_candidates(native, probability, sample, restore, geometry)
                totals = audit_totals[(model['name'], geometry[0], task['condition'])]
                totals['contours'] += audit['contours']
                totals['examined'] += audit['examined']
                totals['candidate_limit_skipped'] += audit['candidate_limit_skipped']
                totals['approximation_rejected'] += audit['approximation_rejected']
                for event in audit['events']:
                    totals['scored'] += 1
                    totals[event['status']] += 1
                matcher = CandidateMatcher(prediction, raw_scores, row, policy)
                for score_i, score in enumerate(SCORES):
                    setting_i = geometry_i*len(SCORES)+score_i
                    scored = matcher.score(score)
                    tp, gt, pred = scored['counts']
                    tiny_gt, tiny_tp = scored['bins'][0]
                    matrix[source_i, condition_index, setting_i] = (tp, gt, pred, tiny_gt, tiny_tp)
                if geometry_i == 0:
                    base_index = geometry_i*len(SCORES)+SCORES.index(.40)
                    baseline_counts = baseline['counts'][:3]
                    require(matrix[source_i, condition_index, base_index, :3].tolist() == baseline_counts,
                            f'Exploratory base `.30/.40` counts differ from historical EDA: {task["sample_id"]}')
                    if source_i == 0:
                        matcher.assert_official(.40)
                elif source_i == 0:
                    matcher.assert_official(.40)
            if number % 500 == 0 or number == len(tasks):
                print(f'{model["name"]}: {number:,}/{len(tasks):,} paired-condition tasks; base `.30/.40` reproduces EDA', flush=True)
        matrices[model['name']] = matrix
        np.savez_compressed(a.output/f'{model["name"]}_source_counts.npz', source_ids=np.asarray(source_ids),
                            settings=np.asarray([json.dumps(x, sort_keys=True) for x in settings]), counts=matrix)
        del detector
        torch.cuda.empty_cache()

    frontier_rows, curve_rows, summaries = [], [], {}
    for model in models:
        name = model['name']
        records = []
        summaries[name] = {}
        for setting_i, setting in enumerate(settings):
            totals = summarize_counts(matrices[name][:, :, setting_i, :])
            record = dict(setting=setting, clean=totals['clean'], lr2=totals['lr2'])
            records.append(record)
            for condition in ('clean', 'lr2'):
                metric = totals[condition]
                row = dict(model=name, setting=setting_key(setting), **setting, condition=condition, **metric)
                frontier_rows.append(row)
                if setting['geometry'] == 'base':
                    curve_rows.append(dict(row, global_fp=metric['fp'], tiny_recall=metric['tiny_recall']))
        selected = choose_setting(records)
        selected_i = settings.index(selected['setting'])
        summaries[name] = dict(selected=selected, all_settings=len(records))
        summaries[name]['matrix_summary'] = dict(clean=summarize_counts(matrices[name][:, :, selected_i, :])['clean'],
                                                  lr2=summarize_counts(matrices[name][:, :, selected_i, :])['lr2'])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(0, len(source_ids), size=(BOOTSTRAP_REPETITIONS, len(source_ids)))
    comparisons = {}
    for model in models:
        name = model['name']
        base_index = COMMON_INDEX
        selected_index = settings.index(summaries[name]['selected']['setting'])
        comparisons[name] = {}
        for condition_index, condition in enumerate(('clean', 'lr2')):
            comparisons[name][condition] = paired_hmean_ci(matrices[name][:, :, base_index, :],
                matrices[name][:, :, selected_index, :], condition_index, bootstrap_indices,
                BOOTSTRAP_REPETITIONS)

    write_csv(result_root/'threshold_frontier.csv',
        ['model','setting','geometry','mask_thr','unclip_ratio','min_text_width','epsilon_ratio','min_text_score',
         'condition','tp','gt','pred','fp','fn','precision','recall','hmean','tiny_gt','tiny_tp','tiny_recall'], frontier_rows)
    write_csv(result_root/'pr_curves.csv',
        ['model','setting','geometry','mask_thr','unclip_ratio','min_text_width','epsilon_ratio','min_text_score',
         'condition','tp','gt','pred','fp','fn','precision','recall','hmean','tiny_gt','tiny_tp','tiny_recall','global_fp'], curve_rows)
    audit_rows = [dict(model=name, geometry=geom, condition=condition, **totals)
                  for (name, geom, condition), totals in sorted(audit_totals.items())]
    write_csv(result_root/'quad_filter_audit.csv',
        ['model','geometry','condition','contours','examined','candidate_limit_skipped','approximation_rejected',
         'scored','unclip_empty','width_rejected','accepted'], audit_rows)

    cases, review_summary = prepare_review(a, models, rows, tasks, policy)
    case_pairs = {(case['source_image_id'], case['condition']) for case in cases}
    review_tasks = [task for task in tasks if (task['source_image_id'], task['condition']) in case_pairs]
    review_model_map = {m['name']: m for m in models}
    for name in ('B-Clean', 'B1'):
        model = review_model_map[name]
        options = copy.copy(a)
        options.checkpoint = model['checkpoint']
        detector, native, _, _ = detector_for(options)
        processor = copy.copy(native)
        processor.mask_thr, processor.min_text_score = .30, .40
        for task in review_tasks:
            row = rows[task['source_image_id']]
            _, probability, sample, factor, restore, _, _ = forward(detector, native, task, row)
            from lowres.inference import unpack
            prediction = unpack(processor(probability, [copy.deepcopy(sample)])[0], restore)
            render_panels(a, model, task, row, probability, factor, prediction, cases)
        del detector
        torch.cuda.empty_cache()
    combine_panels(a, cases)

    summary = dict(identity=identity, sources=len(rows), tasks=len(tasks), models=summaries,
                   selected_ci_vs_original_common=comparisons, review=review_summary,
                   pr_geometry='base quad; eight score thresholds', grid_combinations=len(settings),
                   no_test=True, exploratory=True, human_review='PENDING')
    save_json(result_root/'decision_audit_summary.json', summary)
    report = report_text(a, models, {name: value['selected'] for name, value in summaries.items()},
                         comparisons, review_summary, pp_path)
    (result_root/'decision_audit_report.md').write_text(report, encoding='utf-8')
    save_json(result_root/'complete.json', dict(stage=STAGE, no_test=True, training=False,
                                               report_sha256=sha256(result_root/'decision_audit_report.md'),
                                               identity_sha256=sha256(a.output/'run_identity.json')))
    print('Decision audit complete:', result_root/'decision_audit_report.md', flush=True)
