"""Resume job 54308 from immutable counts; checkpoint review in small chunks."""
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lowres.common import identity as fingerprint, read_json, save_json, sha256
from rev4 import require
from rev4.adaptation_jobs import lock_record
from rev4.calibration import validation_cohort
from rev4.decision_audit import (cached_baseline, installed_postprocessor, model_contexts,
                                report_text)
from rev4.decision_core import choose_setting, paired_hmean_ci, summarize_counts
from rev4.decision_plan import BOOTSTRAP_SEED, COMMON_INDEX, REPETITIONS, REVIEW_SEED, grid
from rev4.decision_review import combine_panels, prepare_review, render_panels
from rev4.stage_a import detector_for, forward

CHUNK_TASKS = 64
REVIEW_FILES = ('fp_taxonomy_proxies.csv', 'gt_diagnosis.csv', 'human_review.csv',
                'selected_review.json', 'diagnosis_summary.json')


def event(a, phase, **details):
    value = dict(phase=phase, **details)
    save_json(a.output/'progress.json', value)
    print(json.dumps(value, sort_keys=True), flush=True)


def load_counts(path, source_ids, settings):
    with np.load(path, allow_pickle=False) as data:
        require(data['source_ids'].tolist() == source_ids, 'Count cache source order changed')
        require([json.loads(s) for s in data['settings']] == settings, 'Count cache grid changed')
        counts = data['counts'].copy()
    require(counts.shape == (len(source_ids), 2, len(settings), 5) and
            np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all(),
            'Count cache shape/dtype/content invalid')
    tp, gt, pred, tiny_gt, tiny_tp = [counts[..., i] for i in range(5)]
    require((tp <= gt).all() and (tp <= pred).all() and (tiny_tp <= tiny_gt).all() and
            (tiny_tp <= tp).all() and (tiny_gt <= gt).all(), 'Inconsistent TP/GT/pred cache')
    return counts


def merged_review(chunks, output):
    """Combine chunk populations and minima: same sample as one full pass."""
    folder = output/'review'
    folder.mkdir(parents=True, exist_ok=True)
    populations, chosen = Counter(), {}
    fp_counts = defaultdict(Counter)
    gt_counts, correspondence = defaultdict(Counter), defaultdict(Counter)
    for chunk in chunks:
        selected = read_json(chunk/'selected_review.json')
        summary = read_json(chunk/'diagnosis_summary.json')
        for item in summary['fp_proxies']:
            fp_counts[item['condition'], item['model']].update(item['counts'])
        for condition, counts in summary['gt_transitions'].items():
            gt_counts[condition].update(counts)
        for condition, counts in summary['fp_correspondence'].items():
            correspondence[condition].update(counts)
        for case in selected['cases']:
            key = tuple(case['stratum'])
            populations[key] += case['stratum_population']
            rank = hashlib.sha256(f'{REVIEW_SEED}:{case["case_id"]}'.encode()).hexdigest()
            if key not in chosen or rank < chosen[key][0]:
                chosen[key] = (rank, dict(case))
    cases = []
    for index, key in enumerate(sorted(chosen)):
        case = chosen[key][1]
        population = populations[key]
        case.update(panel_index=index, image_pair=f'examples/{index:04d}.png',
                    stratum_population=population, inclusion_probability=1/population,
                    population_weight=population, human_label='', reviewer='', notes='')
        cases.append(case)
    for name in ('fp_taxonomy_proxies.csv', 'gt_diagnosis.csv'):
        target = folder/name
        temp = target.with_suffix('.csv.tmp')
        with temp.open('w', encoding='utf-8', newline='') as stream:
            writer = None
            for chunk in chunks:
                with (chunk/name).open(encoding='utf-8', newline='') as source:
                    reader = csv.DictReader(source)
                    if writer is None:
                        writer = csv.DictWriter(stream, reader.fieldnames)
                        writer.writeheader()
                    require(reader.fieldnames == writer.fieldnames, 'Review shard CSV schema changed')
                    writer.writerows(reader)
        os.replace(temp, target)
    fields = ['case_id','condition','group','source_image_id','ann_id','model','prediction_index',
              'height_bin','score','stratum_population','inclusion_probability','population_weight',
              'image_pair','human_label','reviewer','notes']
    with (folder/'human_review.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cases)
    save_json(folder/'selected_review.json', dict(seed=REVIEW_SEED, cases=cases,
        rule='global minimum SHA256 rank per condition/group/height/score stratum',
        manual_review_status='PENDING', invalid_fp_not_rendered=True))
    summary = dict(fp_proxies=[dict(condition=c, model=m, counts=dict(v))
                              for (c, m), v in sorted(fp_counts.items())],
        gt_transitions={c:dict(v) for c,v in gt_counts.items()},
        fp_correspondence={c:dict(v) for c,v in correspondence.items()},
        sampled_cases=len(cases), manual_review_status='PENDING', proxies_are_not_causes=True)
    save_json(folder/'diagnosis_summary.json', summary)
    return cases, summary


def checkpointed_review(a, models, rows, tasks, policy):
    chunks = []
    for offset in range(0, len(tasks), CHUNK_TASKS):
        subset = tasks[offset:offset+CHUNK_TASKS]
        chunk_root = a.output/'review_chunks'/f'{offset//CHUNK_TASKS:04d}'
        folder, receipt = chunk_root/'review', chunk_root/'complete.json'
        key = dict(launch=fingerprint(a.recovery_identity), samples=[t['sample_id'] for t in subset])
        if receipt.is_file():
            previous = read_json(receipt)
            require(previous['identity'] == key and all(sha256(folder/name) == digest
                    for name, digest in previous['files'].items()), 'Review chunk changed after completion')
        else:
            event(a, 'review_chunk_start', offset=offset, first_sample=subset[0]['sample_id'])
            options = SimpleNamespace(output=chunk_root, eda_root=a.eda_root)
            prepare_review(options, models, rows, subset, policy)
            save_json(receipt, dict(identity=key, files={name:sha256(folder/name) for name in REVIEW_FILES}))
        chunks.append(folder)
        event(a, 'review_chunk_complete', completed=min(offset+CHUNK_TASKS,len(tasks)), total=len(tasks))
    return merged_review(chunks, a.output)


def run(a):
    import torch
    from lowres.inference import unpack
    from rev4.b1_training import environment_identity

    require(environment_identity(a, a.calibration_prepared) == a.b1_identity,
            'Recovery runtime differs from historical B1')
    pp_path = installed_postprocessor()
    rows, tasks = validation_cohort(a, a.calibration_prepared)
    require(len(rows) == 2262 and len(tasks) == 4524, 'Expected locked remaining-validation cohort')
    policy = read_json(a.evaluation_protocol)
    models, eda = model_contexts(a)
    original = read_json(a.recovery_source/'run_identity.json')
    require(original['validation_sha256'] == sha256(a.calibration_prepared/'validation_remaining.jsonl') and
            original['task_manifest_sha256'] == sha256(a.calibration_prepared/'validation_tasks.jsonl') and
            original['original_protocol_sha256'] == sha256(a.evaluation_protocol) and
            original['eda_complete_sha256'] == sha256(a.eda_root/'results/B0_BClean_B1/complete.json'),
            'Original sweep data/policy/EDA provenance changed')
    source_ids, settings = sorted(rows), grid()
    require(original['scores'] == list(dict.fromkeys(s['min_text_score'] for s in settings)),
            'Recovery score grid differs from original')
    lock_record(a.output/'run_identity.json', a.recovery_identity)
    result_root = a.output/'results'
    result_root.mkdir(parents=True, exist_ok=True)
    matrices, selected, comparisons = {}, {}, {}
    source_i = {source:i for i,source in enumerate(source_ids)}
    event(a, 'validate_original_counts')
    for model in models:
        name = model['name']
        matrix = load_counts(a.recovery_source/f'{name}_source_counts.npz', source_ids, settings)
        # Revalidate every reused base count against the locked historical EDA cache.
        for task in tasks:
            _, baseline = cached_baseline(a, model, task)
            j = int(task['condition'] == 'lr2')
            require(matrix[source_i[task['source_image_id']], j, COMMON_INDEX, :3].tolist() == baseline['counts'][:3],
                    f'Reused count differs from historical EDA: {name}/{task["sample_id"]}')
        matrices[name] = matrix
        candidates = [dict(setting=setting, **summarize_counts(matrix[:,:,i,:]))
                      for i,setting in enumerate(settings)]
        selected[name] = choose_setting(candidates)
        event(a, 'original_counts_pass', model=name, tasks=len(tasks))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0,len(rows),(REPETITIONS,len(rows)))
    for name, matrix in matrices.items():
        best = settings.index(selected[name]['setting'])
        comparisons[name] = {condition:paired_hmean_ci(matrix[:,:,COMMON_INDEX,:],matrix[:,:,best,:],j,indices,REPETITIONS)
                             for j,condition in enumerate(('clean','lr2'))}
    # Retain the original validated frontier/audit bytes, with hashes in launch identity.
    for name in ('pr_curves.csv','threshold_frontier.csv','quad_filter_audit.csv'):
        content = (a.recovery_source/'results'/name).read_bytes()
        target = result_root/name
        if target.is_file():
            require(target.read_bytes() == content, 'Copied sweep CSV changed')
        else:
            target.write_bytes(content)
    save_json(result_root/'quantitative_summary.json', dict(selected=selected, comparisons=comparisons,
        original_job='54308', original_run_identity_sha256=sha256(a.recovery_source/'run_identity.json'),
        no_test=True, training=False, selection_adjusted_ci=False))
    cases, review_summary = checkpointed_review(a, models, rows, tasks, policy)
    require(review_summary['gt_transitions']['lr2'] == dict(persistent_tp=51028,regressed=1453,
            rescued=6150,persistent_fn=19590), 'Recovered GT transitions differ from historical EDA')
    require(review_summary['fp_correspondence']['lr2'] == dict(persistent_fp=6790,
            older_only_fp=3667,newer_only_fp=9697), 'Recovered FP correspondence differs from historical EDA')
    wanted = {(c['source_image_id'],c['condition']) for c in cases}
    review_tasks = [t for t in tasks if (t['source_image_id'],t['condition']) in wanted]
    for model in [m for m in models if m['name'] in ('B-Clean','B1')]:
        name = model['name']
        missing = [t for t in review_tasks if any(not (a.output/'review/panels'/f'{c["panel_index"]:04d}_{name}.png').is_file()
                   for c in cases if (c['source_image_id'],c['condition']) == (t['source_image_id'],t['condition']))]
        if not missing:
            continue
        options = copy.copy(a)
        options.checkpoint = model['checkpoint']
        detector, native, _, _ = detector_for(options)
        processor = copy.copy(native)
        processor.mask_thr, processor.min_text_score = .30,.40
        for n,task in enumerate(missing,1):
            event(a,'render_sample_start',model=name,sample_id=task['sample_id'],number=n,total=len(missing))
            row = rows[task['source_image_id']]
            _,probability,sample,factor,restore,_,_ = forward(detector,native,task,row)
            prediction = unpack(processor(probability,[copy.deepcopy(sample)])[0],restore)
            _,baseline = cached_baseline(a,model,task)
            require(prediction == baseline['prediction'], 'Review rendering changed historical predictions')
            render_panels(a,model,task,row,probability,factor,prediction,cases)
        del detector
        torch.cuda.empty_cache()
    combine_panels(a,cases)
    summary = dict(identity=a.recovery_identity,selected=selected,comparisons=comparisons,
                   review=review_summary,no_test=True,training=False,human_review='PENDING')
    save_json(result_root/'decision_audit_summary.json',summary)
    report = report_text(a,models,selected,comparisons,review_summary,pp_path)
    report += ('\nRecovery of job 54308: original source-count matrices and sweep CSVs were reused after hash and '
               'per-image baseline checks. Only sampled review images required new forward passes. '
               'Source-paired CIs are descriptive after selection and do not measure training-seed variation.\n')
    (result_root/'decision_audit_report.md').write_text(report,encoding='utf-8')
    save_json(result_root/'complete.json',dict(no_test=True,training=False,human_review='PENDING',
        report_sha256=sha256(result_root/'decision_audit_report.md'),identity=fingerprint(a.recovery_identity)))
    event(a,'complete',report=str(result_root/'decision_audit_report.md'))
