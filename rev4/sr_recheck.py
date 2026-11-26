"""Paired native/bicubic/EDSR inference at both frozen operating points."""
from collections import Counter
from contextlib import contextmanager
import copy
from pathlib import Path
import time

import numpy as np

from lowres.common import identity, read_json, sha256
from rev4 import require
from rev4.durable_io import save_json, write_text
from rev4.fusion_evaluation import materialize, specification, metrics
from rev4.local_scratch import local_directory
from rev4.sr_plan import BRANCHES, PAIRS, PRIMARY, RECIPE

METRICS = ('hmean_pp','precision_pp','recall_pp','fp_count','recall_lt4_pp',
           'recall_4to8_pp','recall_8to16_pp','recall_ge16_pp')


@contextmanager
def resized(a, original, branch, model, device):
    from PIL import Image
    from rev4.edsr import restore
    from rev4.stage_a import open_native_lr, make_task
    if branch == 'native':
        yield original, 0.
        return
    require(branch in BRANCHES, 'Unknown SR branch')
    start = time.perf_counter()
    with open_native_lr(Path(original['path']))[0] as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    size = (2*original['size'][0], 2*original['size'][1])
    if branch == 'edsr_x2':
        restored = restore(model, rgb, device, tile=RECIPE['tile'], halo=RECIPE['halo'])
    else:
        with Image.fromarray(rgb) as image:
            with image.resize(size, Image.Resampling.BICUBIC) as result:
                restored = np.asarray(result).copy()
    elapsed_ms = (time.perf_counter()-start)*1000
    require(restored.dtype == np.uint8 and restored.shape == (size[1], size[0], 3), 'SR output shape/dtype differs')
    with local_directory('monkey-sr-recheck-') as directory:
        path = directory/'restored.png'
        with Image.fromarray(restored) as image:
            image.save(path, format='PNG')
        task = make_task(original['source_image_id'], original['condition'], path, size, sha256(path))
        yield task, elapsed_ms


def infer(detector, native, task, row, policy, official=False):
    from rev4.stage_a import forward
    from rev4.decision_core import capture_candidates, CandidateMatcher
    _, pmap, sample, _, restore, _, elapsed = forward(detector, native, task, row)
    results = []
    for mask, score in PAIRS:
        polygons, raw_scores, _ = capture_candidates(native, pmap, sample, restore, ('fixed', mask, 1.5, 5, .01))
        matcher = CandidateMatcher(polygons, raw_scores, row, policy)
        if official:
            matcher.assert_official(score)
        results.append(matcher.score(score))
    return dict(counts=[v['counts'] for v in results], bins=[v['bins'] for v in results],
                matched=[v['matched_ann_ids'] for v in results], detector_ms=elapsed,
                size=task['size'], image_sha256=task['image_sha256'])


def validate_record(record, task, digest, baseline, baseline_indices):
    require(record['identity_sha256'] == digest and all(record[k] == task[k]
            for k in ('sample_id','source_image_id','condition')), 'SR cached identity/task mismatch')
    require(set(record['branches']) == set(BRANCHES) and record['input_sha256'] == baseline['image_sha256'],
            'SR branches or source input changed')
    base_size = task.get('size',record['branches']['native']['size'])
    require(record['branches']['native']['image_sha256'] == record['input_sha256'], 'Native pixel hash changed')
    for branch, value in record['branches'].items():
        require(len(value['counts']) == len(value['bins']) == len(value['matched']) == len(PAIRS), 'SR operating points differ')
        for counts, bins, matched in zip(value['counts'], value['bins'], value['matched']):
            require(len(counts) == 3 and all(type(v) is int and v >= 0 for v in counts), 'Invalid SR counts')
            tp, gt, pred = counts
            require(tp <= min(gt, pred) and len(set(matched)) == len(matched) == tp, 'SR matched GT/count mismatch')
            require(len(bins) == 4 and all(len(v) == 2 and 0 <= v[1] <= v[0] for v in bins), 'Invalid SR bins')
            require(sum(v[0] for v in bins) == gt and sum(v[1] for v in bins) == tp, 'SR bins/count mismatch')
        require(value['size'] == ([2*v for v in base_size] if branch != 'native' else base_size), 'SR canvas differs')
    for new, old in enumerate(baseline_indices):
        require(record['branches']['native']['counts'][new] == baseline['counts'][old]
                and record['branches']['native']['bins'][new] == baseline['bins'][old], 'Frozen B1 baseline differs per source')


def evaluate_task(a, original, row, detector, native, model, device, policy, official=False):
    record = dict(sample_id=original['sample_id'], source_image_id=original['source_image_id'],
                  condition=original['condition'], input_sha256=original['image_sha256'], branches={})
    for branch in BRANCHES:
        with resized(a, original, branch, model, device) as (task, prep_ms):
            value = infer(detector, native, task, row, policy, official)
        value['restoration_ms'] = prep_ms
        value['total_ms'] = prep_ms + value['detector_ms']
        record['branches'][branch] = value
    return record


def completed_b1_detector(a):
    from rev4.stage_a import detector_for
    require(sha256(a.b1_checkpoint) == a.b1_digest, 'Completed B1 checkpoint changed')
    options = copy.copy(a)
    options.checkpoint = a.b1_checkpoint
    detector, native, _, _ = detector_for(options)
    detector.model.eval().requires_grad_(False)
    native.mask_thr, native.min_text_score = PAIRS[PRIMARY]
    return detector,native


def run(a):
    import torch
    from rev4 import edsr
    rows, tasks, selected = specification(a)
    reference = read_json(a.fusion_reference/'evaluation/B1/counts.json')
    fusion_summary = read_json(a.fusion_reference/'results/fusion_summary.json')
    baseline_indices = [fusion_summary['threshold_pairs'].index(list(pair)) for pair in PAIRS]
    require(reference['checkpoint_sha256'] == a.b1_digest and len(tasks) == len(reference['tasks']) == 7524,
            'Expected completed B1 reference')
    require([v['sample_id'] for v in tasks] == [v['sample_id'] for v in reference['tasks']], 'SR task order differs')
    require(sha256(a.edsr_weight) == a.trial_identity['edsr_sha256'], 'EDSR weights changed')
    detector,native = completed_b1_detector(a)
    model = edsr.EDSR().float().eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.edsr_weight, map_location='cpu', weights_only=True), strict=True)
    device = torch.device('cuda:0'); model.to(device)
    require(all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters()), 'Require frozen FP32 EDSR')
    policy = read_json(a.evaluation_protocol); digest = identity(a.trial_identity)
    historical = {v['source_image_id']:v for v in read_json(a.old_restoration/'paired_records.json')['records']}
    # Real GPU smoke test and baseline parity for every condition, before the full pass.
    probe_file = a.output/'preflight.json'
    if not probe_file.exists():
        probes = []; seen = set(); historical_checks = []
        for task, baseline in zip(tasks, reference['tasks']):
            condition = task['condition']
            if condition in seen:
                continue
            row = rows[task['source_image_id']]
            with materialize(a, task, row) as current:
                record = evaluate_task(a, current, row, detector, native, model, device, policy, True)
            record['identity_sha256'] = digest
            validate_record(record, task, digest, baseline, baseline_indices)
            if condition == 'lr2':
                prior = historical[task['source_image_id']]['branches']
                for branch, old_branch in zip(BRANCHES, ('lr2','bicubic_x2','edsr_x2')):
                    now, old = record['branches'][branch], prior[old_branch]
                    historical_checks.append(dict(branch=branch, source_image_id=task['source_image_id'],
                        old_pair_counts_match=now['counts'][0] == old['counts'][:3],
                        old_pair_matched_ids_match=set(now['matched'][0]) == set(old['matched'])))
            probes.append(dict(sample_id=task['sample_id'], native_matches=True, indexed_matches_official=True))
            seen.add(condition)
        require(len(seen) == 8, 'Incomplete SR smoke test')
        save_json(probe_file, dict(status='PASS', identity=a.trial_identity, probes=probes,
                                  historical_lr2_checks=historical_checks, no_training=True))
        print('PASS EDSR/B1 GPU smoke test across 8 conditions, both frozen operating points', flush=True)
    else:
        require(read_json(probe_file)['identity'] == a.trial_identity and read_json(probe_file)['status']=='PASS', 'SR probe changed')
    checked = set()
    for number, (task, baseline) in enumerate(zip(tasks, reference['tasks']), 1):
        sid = task['source_image_id']; row = rows[sid]
        if sid not in checked:
            require(sha256(a.clean_root/row['clean_relpath']) == row['clean_sha256'], 'Clean source changed')
            checked.add(sid)
        if 'path' in task:
            require(sha256(task['path']) == task['image_sha256'], 'Prepared input changed')
        target = a.output/'records'/task['condition']/(sid+'.json')
        if target.exists():
            validate_record(read_json(target), task, digest, baseline, baseline_indices)
        else:
            with materialize(a, task, row) as current:
                record = evaluate_task(a, current, row, detector, native, model, device, policy)
            record['identity_sha256'] = digest
            validate_record(record, task, digest, baseline, baseline_indices)
            save_json(target, record)
        if number % 50 == 0 or number == len(tasks):
            save_json(a.output/'progress.json', dict(identity_sha256=digest, completed_tasks=number, total_tasks=len(tasks)))
            print('EDSR recheck', number, '/', len(tasks), 'x3 branches', flush=True)
    summarize(a, tasks, selected, digest)


def metric_vector(totals):
    """TP,GT,pred followed by four GT/TP pairs; paired source bootstrap endpoints."""
    x = np.asarray(totals); tp, gt, pred = x[...,0], x[...,1], x[...,2]
    return np.stack([100*2*tp/np.maximum(gt+pred,1),100*tp/np.maximum(pred,1),100*tp/np.maximum(gt,1),pred-tp]
                    +[100*x[...,4+2*i]/np.maximum(x[...,3+2*i],1) for i in range(4)],axis=-1)


def bootstrap_group(matrices, repetitions=2000, seed=20261107):
    arrays = {key:np.asarray(value,dtype=np.int64) for key,value in matrices.items()}
    size = len(arrays['native'])
    require(size > 0 and all(v.shape == (size,11) for v in arrays.values()), 'Unpaired SR bootstrap matrices')
    rng = np.random.default_rng(seed); samples = {key:[] for key in arrays}
    for start in range(0,repetitions,50):
        indices = rng.integers(0,size,size=(min(50,repetitions-start),size))
        for key,value in arrays.items(): samples[key].append(metric_vector(value[indices].sum(1)))
    samples = {key:np.concatenate(value) for key,value in samples.items()}; result = {}
    for new,old in (('edsr_x2','native'),('bicubic_x2','native'),('edsr_x2','bicubic_x2')):
        delta = metric_vector(arrays[new].sum(0))-metric_vector(arrays[old].sum(0))
        ci = np.quantile(samples[new]-samples[old],[.025,.975],axis=0)
        result[new+'_minus_'+old] = {name:dict(delta=float(delta[i]),ci95=[float(ci[0,i]),float(ci[1,i])])
                                    for i,name in enumerate(METRICS)}
    return result


def summarize(a,tasks,selected,digest):
    records = []; cohort = set(selected)
    for task in tasks:
        records.append(read_json(a.output/'records'/task['condition']/(task['source_image_id']+'.json')))
    scores = {}; comparisons = {}; transitions = {}; timing = {}
    groups = list(RECIPE['conditions'])+['anchor_clean','anchor_lr2']
    for pair_index,pair in enumerate(PAIRS):
        pair_name = f'{pair[0]:.2f}/{pair[1]:.2f}'
        scores[pair_name] = {}; comparisons[pair_name] = {}; transitions[pair_name] = {}
        for condition in groups:
            anchor = condition.startswith('anchor_'); actual = condition.removeprefix('anchor_')
            group = sorted((r for r in records if r['condition']==actual and
                            (not anchor or r['source_image_id'] in cohort)), key=lambda r:r['source_image_id'])
            require(len(group)==(2262 if actual in ('clean','lr2') and not anchor else 500),'SR report cohort mismatch')
            matrices = {}; scores[pair_name][condition] = {}; transitions[pair_name][condition] = {}
            for branch in BRANCHES:
                matrix = np.asarray([r['branches'][branch]['counts'][pair_index]+
                                     sum(r['branches'][branch]['bins'][pair_index],[]) for r in group],dtype=np.int64)
                matrices[branch] = matrix
                total = matrix.sum(0); score = metrics(total[:3])
                score['bins_lr2_reference'] = total[3:].reshape(4,2).tolist()
                scores[pair_name][condition][branch] = score
                timing.setdefault(condition,{})[branch] = dict(
                    median_total_ms=float(np.median([r['branches'][branch]['total_ms'] for r in group])),
                    median_restoration_ms=float(np.median([r['branches'][branch]['restoration_ms'] for r in group])),
                    images=len(group),scope=RECIPE['timing'])
                if branch != 'native':
                    gained = lost = 0
                    for record in group:
                        before = set(record['branches']['native']['matched'][pair_index])
                        after = set(record['branches'][branch]['matched'][pair_index])
                        gained += len(after-before); lost += len(before-after)
                    transitions[pair_name][condition][branch] = dict(rescued_gt=gained,lost_gt=lost)
            comparisons[pair_name][condition] = bootstrap_group(matrices)
    # Compare the complete historical LR2 common operating point without overwriting it.
    old = read_json(a.old_restoration/'restoration_summary.json')
    historical = {branch:dict(new_hmean=scores['0.30/0.40']['lr2'][branch]['hmean'],
                    old_hmean=old['scores'][prior]['hmean'])
                  for branch,prior in zip(BRANCHES,('lr2','bicubic_x2','edsr_x2'))}
    output = a.output/'results'; output.mkdir(exist_ok=True)
    summary = dict(identity=a.trial_identity,recipe=RECIPE,scores=scores,comparisons=comparisons,
        transitions=transitions,timing=timing,historical_common_point=historical,no_test=True,no_training=True,
        notes=['all development already viewed; one detector seed',
               'multiple endpoints exploratory, unadjusted source CIs do not measure seed/selection uncertainty',
               'EDSR pretrained bicubic degradation overlaps synthetic LR; not real-LR validation',
               'always-on EDSR includes Clean; no oracle corruption selector'])
    save_json(output/'edsr_recheck_summary.json',summary)
    lines=['# EDSR before completed B1 · fixed operating-point recheck','',
           'No training. Same completed B1 weights in every branch. Native, bicubic x2 and pretrained EDSR x2.',
           '2262 paired Clean/LR2 sources; same500-source corruption cohort as fusion111683. No TEST.',
           'Always-on restoration, including Clean; bins stay tied to original native LR2 GT heights.','']
    for pair in PAIRS:
        key=f'{pair[0]:.2f}/{pair[1]:.2f}'
        lines += [f'## Fixed mask/score {key}'+(' (primary)' if pair==PAIRS[PRIMARY] else ' (historical)'),'',
                  '| Condition | Input | P | R | Hmean | TP | FP | FN | Recall <4px | Median ms |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for condition in RECIPE['conditions']:
            for branch in BRANCHES:
                s=scores[key][condition][branch];gt,tp=s['bins_lr2_reference'][0]
                lines.append(f'| {condition} | {branch} | {s["precision"]:.2%} | {s["recall"]:.2%} | {s["hmean"]:.2%} | '
                             f'{s["tp"]} | {s["fp"]} | {s["fn"]} | {tp/max(gt,1):.2%} | {timing[condition][branch]["median_total_ms"]:.1f} |')
        lines += ['', '| Condition | Comparison | Delta H pp (95%CI) | Delta tiny recall pp (95%CI) |',
                  '|---|---|---:|---:|']
        for condition in ('clean','lr2'):
            for name,value in comparisons[key][condition].items():
                h,t=value['hmean_pp'],value['recall_lt4_pp']
                lines.append(f'| {condition} | {name} | {h["delta"]:+.3f} {h["ci95"]} | {t["delta"]:+.3f} {t["ci95"]} |')
    lines += ['', 'Use anchor_clean/anchor_lr2 in JSON for fair500-source stress comparisons. '
              'JSON includes all4 size bins, P/R/FP CIs, rescued/lost GT and historical reproduction.',
              'Timing is non-isolated, excludes PNG encoding/hashes/scoring; does not establish production throughput.',
              'No retuning, TEST, training or automated method selection. Source CIs are exploratory and do not cover training-seed uncertainty.','']
    write_text(output/'edsr_recheck_report.md','\n'.join(lines))
    save_json(output/'complete.json',dict(identity=a.trial_identity,no_test=True,no_training=True,
        report_sha256=sha256(output/'edsr_recheck_report.md'),summary_sha256=sha256(output/'edsr_recheck_summary.json')))
