"""Inference-only residual diagnosis at the already selected .40/.60 setting."""
from collections import Counter, defaultdict
import copy
import csv
from pathlib import Path

import numpy as np

from lowres.common import identity as fingerprint, read_json, save_json, sha256
from lowres.evaluate_shared import bootstrap_paired, rates
from rev4 import require
from rev4.adaptation_jobs import lock_record
from rev4.adaptation_plan import HEIGHT_NAMES
from rev4.calibration import validation_cohort
from rev4.decision_audit import cached_baseline, installed_postprocessor, model_contexts
from rev4.decision_plan import BOOTSTRAP_SEED, REPETITIONS, grid
from rev4.decision_recovery import checkpointed_review, event, load_counts
from rev4.fixed_diagnostic_core import (MASK, PAIR_KEY, SCORE, assert_reference,
                                        instance_transitions, kernel_at_threshold)
from rev4.fixed_review import combine, render, valid_panel
from rev4.paired_eda import mode_result
from rev4.paired_eda_core import instance_evidence
from rev4.stage_a import detector_for, forward


def record_path(a,name,task):
    return a.output/'models'/name/task['condition']/f'{task["source_image_id"]}.json'


def fixed_result(record):
    return record['results'][PAIR_KEY]


def infer(a,model,rows,tasks,reference,source_index):
    import torch
    from mmocr.utils import offset_polygon
    from lowres.inference import unpack

    name=model['name']
    cache_model=dict(name=name,checkpoint_sha256=model['checkpoint_sha256'],
                     postprocessor=dict(mask_thr=MASK,min_text_score=SCORE,text_repr_type='quad',
                                        unclip_ratio=1.5,min_text_width=5,epsilon_ratio=.01,max_candidates=3000))
    identity=lock_record(a.output/'models'/name/'identity.json',
        dict(model=cache_model,run=fingerprint(a.recovery_identity)))
    digest=fingerprint(identity)
    detector=None
    try:
        for number,task in enumerate(tasks,1):
            source,condition=task['source_image_id'],task['condition']
            expected=reference[source_index[source],int(condition=='lr2')]
            historical,old=cached_baseline(a,model,task)
            target=record_path(a,name,task)
            if target.is_file():
                record=read_json(target)
                require(record['identity_sha256']==digest and record['sample_id']==task['sample_id'] and
                        record['image_sha256']==task['image_sha256'],'Fixed diagnostic cache identity changed')
            else:
                if detector is None:
                    options=copy.copy(a)
                    options.checkpoint=model['checkpoint']
                    detector,native,_,shrink=detector_for(options)
                    processor,common,candidate=copy.copy(native),copy.copy(native),copy.copy(native)
                    processor.mask_thr,processor.min_text_score=MASK,SCORE
                    common.mask_thr,common.min_text_score=.30,.40
                    candidate.mask_thr,candidate.min_text_score=MASK,0.
                event(a,'infer_sample_start',model=name,sample_id=task['sample_id'],completed=number-1,total=len(tasks))
                row=rows[source]
                processed,probability,sample,factor,restore,prediction,_=forward(detector,processor,task,row)
                require(unpack(common(probability,[copy.deepcopy(sample)])[0],restore)==old['prediction'],
                        f'Historical .30/.40 predictions changed: {name}/{task["sample_id"]}')
                img_shape=tuple(sample.metainfo['img_shape'][:2])
                dimensions=dict(native_size=task['size'],resized_hw=list(img_shape),
                                padded_input=list(processed['inputs'].shape),pmap=list(probability.shape))
                require(all(dimensions[k]==historical['dimensions'][k] for k in dimensions),
                        'Image/P-map preprocessing grid changed')
                kernel,_,_=kernel_at_threshold(probability[0].detach().cpu().numpy(),img_shape,
                                               factor,task,row,task['size'],shrink,offset_polygon)
                precutoff=unpack(candidate(probability,[copy.deepcopy(sample)])[0],restore)
                result=instance_evidence(prediction,precutoff,row,a.policy,kernel,SCORE)
                record=dict(identity_sha256=digest,sample_id=task['sample_id'],source_image_id=source,
                    condition=condition,image_sha256=task['image_sha256'],
                    # "common" means the same fixed setting for the two models in THIS cache.
                    modes=dict(fixed=PAIR_KEY,common=PAIR_KEY),results={PAIR_KEY:result},
                    gt_attributes=historical['gt_attributes'],kernel_evidence=kernel,
                    kernel_mask_threshold=MASK,weak_kernel_fraction=.30,
                    historical_kernel_mask_threshold=.30,dimensions=dimensions)
                assert_reference(result,record['gt_attributes'],expected)
                save_json(target,record)
            assert_reference(fixed_result(record),record['gt_attributes'],expected)
            if number%250==0 or number==len(tasks):
                event(a,'infer_counts_pass',model=name,completed=number,total=len(tasks))
    finally:
        if detector is not None:
            del detector
        torch.cuda.empty_cache()
    return dict(model,eda_model=cache_model)


def score(counts):
    tp,gt,pred,covered=map(int,counts)
    return dict(**rates(counts),tp=tp,gt=gt,pred=pred,fp=pred-tp,fn=gt-tp)


def summarize(a,models,rows,tasks):
    output=a.output/'results'
    output.mkdir(parents=True,exist_ok=True)
    source_ids=sorted(rows)
    positions={s:i for i,s in enumerate(source_ids)}
    matrices={(m['name'],c):np.zeros((len(rows),4),dtype=np.int64) for m in models for c in ('clean','lr2')}
    old_matrices={k:np.zeros_like(v) for k,v in matrices.items()}
    postprocess=defaultdict(Counter)
    strata=defaultdict(lambda:np.zeros(2,dtype=np.int64))
    diagnoses=defaultdict(Counter)
    distributions=defaultdict(list)
    fields=['model','source_id','condition','ann_id','height_bin','transition','old_tp','new_tp',
            'old_candidate_iou','new_candidate_iou','new_candidate_score','new_score_margin',
            'new_kernel_coverage_p_gt_040','new_diagnosis']
    with (output/'postprocess_gt_transitions.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fields)
        writer.writeheader()
        for number,task in enumerate(tasks,1):
            source,condition=task['source_image_id'],task['condition']
            for model in models:
                name=model['name']
                record=read_json(record_path(a,name,task)); result=fixed_result(record)
                _,old=cached_baseline(a,model,task)
                matrices[name,condition][positions[source]]=result['counts']
                old_matrices[name,condition][positions[source]]=old['counts']
                postprocess[name,condition].update(instance_transitions(old,result))
                for ann,item in result['instances'].items():
                    attributes=record['gt_attributes'][ann]
                    strata[name,condition,attributes['height_bin']]+=[1,int(item['tp'])]
                    diagnoses[name,condition].update([item['diagnosis']])
                    previous=old['instances'][ann]
                    from rev4.paired_eda_core import transition
                    writer.writerow(dict(model=name,source_id=source,condition=condition,ann_id=ann,
                        height_bin=attributes['height_bin'],transition=transition(previous['tp'],item['tp']),
                        old_tp=previous['tp'],new_tp=item['tp'],old_candidate_iou=previous['candidate_iou'],
                        new_candidate_iou=item['candidate_iou'],new_candidate_score=item['candidate_score'],
                        new_score_margin=item['candidate_score_margin'],
                        new_kernel_coverage_p_gt_040=record['kernel_evidence'][ann]['coverage'],new_diagnosis=item['diagnosis']))
                    if not item['tp']:
                        for key in ('candidate_iou','candidate_score_margin'):
                            if item[key] is not None:
                                distributions[name,condition,item['diagnosis'],key].append(item[key])
            if number%500==0:
                event(a,'summarize_records',completed=number,total=len(tasks))
    scores={name:{c:score(matrices[name,c].sum(axis=0)) for c in ('clean','lr2')} for name in ('B-Clean','B1')}
    comparisons={c:bootstrap_paired(np.concatenate([matrices['B-Clean',c],matrices['B1',c]],axis=1),REPETITIONS,BOOTSTRAP_SEED)
                 for c in ('clean','lr2')}
    pp_comparisons={name:{c:bootstrap_paired(np.concatenate([old_matrices[name,c],matrices[name,c]],axis=1),REPETITIONS,BOOTSTRAP_SEED)
                         for c in ('clean','lr2')} for name in scores}
    with (output/'stratified_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,['model','condition','height_bin','gt','tp','recall'])
        writer.writeheader()
        for (name,condition,bucket),(gt,tp) in sorted(strata.items()):
            writer.writerow(dict(model=name,condition=condition,height_bin=bucket,gt=int(gt),tp=int(tp),recall=float(tp/gt) if gt else None))
    diagnostic=dict(diagnoses=[dict(model=n,condition=c,counts=dict(v)) for (n,c),v in sorted(diagnoses.items())],
        distributions=[dict(model=n,condition=c,diagnosis=d,metric=k,n=len(v),
                       quantiles=dict(zip(('min','q25','median','q75','max'),np.quantile(v,[0,.25,.5,.75,1]).tolist())))
                       for (n,c,d,k),v in sorted(distributions.items())],
        kernel_mask_threshold=MASK,weak_kernel_fraction=.30,proxy_not_visual_cause=True)
    save_json(output/'failure_distributions.json',diagnostic)
    np.savez_compressed(output/'paired_source_counts.npz',source_ids=np.asarray(source_ids),
                        **{name.replace('-','_')+'_'+c:matrix for (name,c),matrix in matrices.items()})
    return dict(scores=scores,comparison_B1_minus_BClean=comparisons,postprocess_comparison=pp_comparisons,
                postprocess_gt_transitions=[dict(model=n,condition=c,counts=dict(v)) for (n,c),v in sorted(postprocess.items())])


def run(a):
    import torch
    from lowres.inference import unpack
    from rev4.b1_training import environment_identity

    require(environment_identity(a,a.calibration_prepared)==a.b1_identity,'Historical GPU/runtime mismatch')
    installed_postprocessor()
    rows,tasks=validation_cohort(a,a.calibration_prepared)
    require(len(rows)==2262 and len(tasks)==4524,'Unexpected remaining-validation cohort')
    models,_=model_contexts(a)
    models=[m for m in models if m['name'] in ('B-Clean','B1')]
    a.policy=read_json(a.evaluation_protocol)
    lock_record(a.output/'run_identity.json',a.recovery_identity)
    source_ids=sorted(rows)
    positions={s:i for i,s in enumerate(source_ids)}
    settings=grid()
    fixed_index=next(i for i,s in enumerate(settings) if s['geometry']=='mask_040' and s['min_text_score']==SCORE)
    review_models=[]
    for model in models:
        reference=load_counts(a.recovery_source/f'{model["name"]}_source_counts.npz',source_ids,settings)[:,:,fixed_index,:]
        review_models.append(infer(a,model,rows,tasks,reference,positions))
    result=summarize(a,models,rows,tasks)
    # Reuse the checkpointed taxonomy machinery against THIS cache's common .40/.60.
    options=copy.copy(a)
    options.eda_root=a.output
    cases,review=checkpointed_review(options,review_models,rows,tasks,a.policy)
    wanted={(c['source_image_id'],c['condition']) for c in cases}
    selected_tasks=[t for t in tasks if (t['source_image_id'],t['condition']) in wanted]
    clean_tasks={t['source_image_id']:t for t in tasks if t['condition']=='clean'}
    for model in models:
        name=model['name']
        missing=[t for t in selected_tasks if any(not valid_panel(a.output/'review/panels'/f'{c["panel_index"]:04d}_{name}.png')
            for c in cases if (c['source_image_id'],c['condition'])==(t['source_image_id'],t['condition']))]
        if not missing:
            continue
        config=copy.copy(a)
        config.checkpoint=model['checkpoint']
        detector,native,_,_=detector_for(config)
        processor=copy.copy(native)
        processor.mask_thr,processor.min_text_score=MASK,SCORE
        for task in missing:
            event(a,'render_sample_start',model=name,sample_id=task['sample_id'])
            row=rows[task['source_image_id']]
            _,probability,sample,factor,restore,_,_=forward(detector,processor,task,row)
            prediction=unpack(processor(probability,[copy.deepcopy(sample)])[0],restore)
            cached=read_json(record_path(a,name,task))
            require(prediction==fixed_result(cached)['prediction'],'Review forward differs from cached fixed prediction')
            render(a,model,task,clean_tasks[task['source_image_id']],row,probability,factor,prediction,cases)
        del detector
        torch.cuda.empty_cache()
    combine(a,cases)
    summary=dict(identity=a.recovery_identity,sources=len(rows),tasks=len(tasks),review=review,
                 human_review='PENDING',no_test=True,training=False,threshold_sweep=False,**result)
    output=a.output/'results'
    save_json(output/'fixed_diagnostics_summary.json',summary)
    lines=['# Fixed .40/.60 residual diagnostics','',
        '2,262 repeatedly used development sources; paired Clean/native LR2. No threshold sweep or training. TEST not read.',
        'Both models use mask=0.40, score=0.60, quad, unclip=1.5, min width=5, epsilon=.01, FP32.',
        'Every image reproduced the historical .30/.40 prediction and the selected sweep .40/.60 TP/GT/pred/tiny counts.','',
        '| Model | Condition | P | R | Hmean | TP | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,conditions in result['scores'].items():
        for condition,v in conditions.items():
            lines.append(f'| {name} | {condition} | {v["precision"]:.2%} | {v["recall"]:.2%} | {v["hmean"]:.2%} | {v["tp"]} | {v["fp"]} | {v["fn"]} |')
    lines+=['','## B1 minus B-Clean','']
    for condition,comparison in result['comparison_B1_minus_BClean'].items():
        delta=100*(result['scores']['B1'][condition]['hmean']-result['scores']['B-Clean'][condition]['hmean'])
        lines.append(f'- {condition}: {delta:+.3f} pp; paired source 95% CI {comparison["delta_pp_95ci"]["hmean"]}.')
    lines+=['','## Diagnosis and review','',
        f'{len(cases)} paired review cards; human review PENDING. Sampling is one deterministic minimum-hash case per '
        'condition/group/height/score stratum; inclusion probabilities and population weights are recorded. '
        'This small stratified sample is for diagnosis, not a precise prevalence estimate.',
        'Unlike the historical review, all predictions/FP correspondence here use .40/.60. '
        'Kernel coverage uses P-map >0.40; weak-response proxy means <30% of valid kernel pixels pass this cutoff. '
        'These proxies are not demonstrated causes, and unmatched text is not automatically background.',
        'Inspect review/examples with GT and P-map before choosing a feature or localization intervention. '
        'No adapter is automatically selected or trained by this job.','',
        'Artifacts: failure_distributions.json, stratified_metrics.csv, postprocess_gt_transitions.csv, '
        'paired_source_counts.npz, review/diagnosis_summary.json, review/human_review.csv and review/examples/.',
        'CIs are conditional source bootstrap intervals after development selection, not training-seed variability.','']
    (output/'fixed_diagnostics_report.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(output/'complete.json',dict(identity=fingerprint(a.recovery_identity),no_test=True,training=False,
              threshold_sweep=False,human_review='PENDING',report_sha256=sha256(output/'fixed_diagnostics_report.md')))
    event(a,'complete',report=str(output/'fixed_diagnostics_report.md'))
