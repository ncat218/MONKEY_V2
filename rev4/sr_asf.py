"""Run only missing ASF cells; import completed B1 controls with pixel/hash checks."""
import copy
from pathlib import Path

import numpy as np

from lowres.common import read_json,sha256,identity
from rev4 import require
from rev4.durable_io import save_json,write_text
from rev4.fusion_evaluation import specification,materialize,configuration,metrics
from rev4.sr_plan import BRANCHES,PAIRS,PRIMARY
from rev4.sr_recheck import evaluate_task,validate_record,metric_vector,METRICS
from rev4.sr_asf_plan import RECIPE,CELLS,COMPARISONS


def completed_asf_detector(a):
    from rev4.stage_a import detector_for
    require(sha256(a.asf_checkpoint)==a.asf_digest,'Completed ASF checkpoint changed')
    path=a.output/'asf_inference_config.py'
    write_text(path,configuration(a,'asf').pretty_text)
    options=copy.copy(a);options.model_config=path;options.checkpoint=a.asf_checkpoint
    detector,native,_,_=detector_for(options)
    require(detector.model.neck.mode=='asf' and detector.model.neck.asf_cfg is not None,'Wrong detector architecture')
    detector.model.eval().requires_grad_(False)
    native.mask_thr,native.min_text_score=PAIRS[PRIMARY]
    return detector,native


def same_inputs(asf,b1):
    require(asf['sample_id']==b1['sample_id'] and asf['input_sha256']==b1['input_sha256'],'Unpaired ASF/B1 source')
    for branch in BRANCHES:
        a,b=asf['branches'][branch],b1['branches'][branch]
        require(a['size']==b['size'] and a['image_sha256']==b['image_sha256'],
                'ASF/B1 restoration pixels differ: '+branch+' '+asf['sample_id'])


def b1_record(a,task,baseline,indices):
    path=a.sr_reference/'records'/task['condition']/(task['source_image_id']+'.json')
    value=read_json(path)
    validate_record(value,task,a.sr_identity_sha,baseline,indices)
    return value,sha256(path)


def run(a):
    import torch
    from rev4.edsr import EDSR
    rows,tasks,selected=specification(a)
    fusion_summary=read_json(a.fusion_reference/'results/fusion_summary.json')
    indices=[fusion_summary['threshold_pairs'].index(list(v)) for v in PAIRS]
    references={name:read_json(a.fusion_reference/f'evaluation/{name}/counts.json') for name in ('B1','asf')}
    for name,record in references.items():
        require(record['checkpoint_sha256']==(a.b1_digest if name=='B1' else a.asf_digest),'Wrong native reference checkpoint')
        require([v['sample_id'] for v in record['tasks']]==[v['sample_id'] for v in tasks] and len(tasks)==7524,
                'Combination task order/coverage changed')
    detector,native=completed_asf_detector(a)
    sr=EDSR().float().eval().requires_grad_(False)
    sr.load_state_dict(torch.load(a.edsr_weight,map_location='cpu',weights_only=True),strict=True)
    device=torch.device('cuda:0');sr.to(device)
    policy=read_json(a.evaluation_protocol);digest=identity(a.trial_identity)
    probe=a.output/'preflight.json'
    if not probe.exists():
        seen=set();checked=[]
        for task,base_asf,base_b1 in zip(tasks,references['asf']['tasks'],references['B1']['tasks']):
            if task['condition'] in seen:continue
            parent,_=b1_record(a,task,base_b1,indices)
            with materialize(a,task,rows[task['source_image_id']]) as current:
                value=evaluate_task(a,current,rows[task['source_image_id']],detector,native,sr,device,policy,True)
            value['identity_sha256']=digest
            validate_record(value,task,digest,base_asf,indices);same_inputs(value,parent)
            seen.add(task['condition']);checked.append(dict(sample_id=task['sample_id'],native_asf_parity=True,
                restored_bytes_equal_b1=True,official_matching_parity=True))
        require(len(seen)==8,'Incomplete ASF+SR GPU probe')
        save_json(probe,dict(status='PASS',identity=a.trial_identity,probes=checked,no_test=True,no_training=True))
        print('PASS ASF+EDSR probe: eight conditions, exact restored pixels, both fixed thresholds',flush=True)
    else:require(read_json(probe)['status']=='PASS' and read_json(probe)['identity']==a.trial_identity,'ASF+SR probe changed')
    checked=set()
    for number,(task,base_asf,base_b1) in enumerate(zip(tasks,references['asf']['tasks'],references['B1']['tasks']),1):
        sid=task['source_image_id'];row=rows[sid]
        if sid not in checked:
            require(sha256(a.clean_root/row['clean_relpath'])==row['clean_sha256'],'Clean source changed');checked.add(sid)
        if 'path' in task:require(sha256(task['path'])==task['image_sha256'],'Prepared input changed')
        parent,parent_sha=b1_record(a,task,base_b1,indices)
        target=a.output/'records'/task['condition']/(sid+'.json')
        if target.exists():
            value=read_json(target);validate_record(value,task,digest,base_asf,indices)
            require(value['b1_record_sha256']==parent_sha,'Completed B1 SR record changed')
        else:
            with materialize(a,task,row) as current:
                value=evaluate_task(a,current,row,detector,native,sr,device,policy)
            value.update(identity_sha256=digest,b1_record_sha256=parent_sha)
            validate_record(value,task,digest,base_asf,indices);same_inputs(value,parent);save_json(target,value)
        same_inputs(value,parent)
        if number%50==0 or number==len(tasks):
            save_json(a.output/'progress.json',dict(identity_sha256=digest,completed_tasks=number,total_tasks=len(tasks)))
            print('EDSR+ASF',number,'/',len(tasks),'x3 new ASF branches; completed B1 controls reused',flush=True)
    summarize(a,tasks,selected)


def paired_effects(matrices,repetitions=2000,seed=20261107):
    arrays={key:np.asarray(value,dtype=np.int64) for key,value in matrices.items()}
    require(set(arrays)==set(CELLS),'Incomplete factorial cells')
    n=len(arrays['B1/native'])
    require(n>0 and all(v.shape==(n,11) for v in arrays.values()),'Unpaired combination matrices')
    rng=np.random.default_rng(seed);draws={key:[] for key in arrays}
    for start in range(0,repetitions,50):
        take=rng.integers(0,n,size=(min(50,repetitions-start),n))
        for key,value in arrays.items():draws[key].append(metric_vector(value[take].sum(1)))
    samples={key:np.concatenate(value) for key,value in draws.items()}
    points={key:metric_vector(value.sum(0)) for key,value in arrays.items()}
    def result(point,sample):
        ci=np.quantile(sample,[.025,.975],axis=0)
        return {name:dict(delta=float(point[i]),ci95=[float(ci[0,i]),float(ci[1,i])]) for i,name in enumerate(METRICS)}
    effects={new+'_minus_'+old:result(points[new]-points[old],samples[new]-samples[old]) for new,old in COMPARISONS}
    effects['interaction']=result((points['ASF/edsr_x2']-points['ASF/native'])-(points['B1/edsr_x2']-points['B1/native']),
        (samples['ASF/edsr_x2']-samples['ASF/native'])-(samples['B1/edsr_x2']-samples['B1/native']))
    return effects


def summarize(a,tasks,selected):
    paired=[];cohort=set(selected);digest=identity(a.trial_identity)
    for task in tasks:
        asf=read_json(a.output/'records'/task['condition']/(task['source_image_id']+'.json'))
        b1_path=a.sr_reference/'records'/task['condition']/(task['source_image_id']+'.json');b1=read_json(b1_path)
        require(asf['identity_sha256']==digest and b1['identity_sha256']==a.sr_identity_sha
                and asf['b1_record_sha256']==sha256(b1_path),'Combination records changed before report')
        same_inputs(asf,b1);paired.append((task,b1,asf))
    parent_summary=read_json(a.sr_reference/'results/edsr_recheck_summary.json')
    scores={};comparisons={};timing={};transitions={}
    for k,pair in enumerate(PAIRS):
        key=f'{pair[0]:.2f}/{pair[1]:.2f}'
        scores[key]={};comparisons[key]={};transitions[key]={}
        for condition in list(RECIPE['conditions'])+['anchor_clean','anchor_lr2']:
            actual=condition.removeprefix('anchor_');anchor=condition.startswith('anchor_')
            group=sorted((v for v in paired if v[0]['condition']==actual and (not anchor or v[0]['source_image_id'] in cohort)),
                         key=lambda v:v[0]['source_image_id'])
            require(len(group)==(2262 if actual in ('clean','lr2') and not anchor else 500),'Combination cohort changed')
            matrices={};scores[key][condition]={};transitions[key][condition]={};timing.setdefault(condition,{})
            for detector,index in (('B1',1),('ASF',2)):
                for branch in BRANCHES:
                    cell=detector+'/'+branch
                    matrix=np.asarray([v[index]['branches'][branch]['counts'][k]+sum(v[index]['branches'][branch]['bins'][k],[]) for v in group])
                    matrices[cell]=matrix;total=matrix.sum(0);score=metrics(total[:3]);score['bins_lr2_reference']=total[3:].reshape(4,2).tolist()
                    scores[key][condition][cell]=score
                    if detector=='B1':require(score==parent_summary['scores'][key][condition][branch],'Imported B1 aggregate differs')
                    timing[condition][cell]=dict(median_total_ms=float(np.median([v[index]['branches'][branch]['total_ms'] for v in group])),
                        scope='non-isolated; B1 from baseline163012 allocation, ASF from this allocation; cross-run descriptive')
                    gained=lost=0
                    for _,b1,asf in group:
                        before=set(b1['branches']['native']['matched'][k]);after=set((b1 if detector=='B1' else asf)['branches'][branch]['matched'][k])
                        gained+=len(after-before);lost+=len(before-after)
                    transitions[key][condition][cell]=dict(rescued_vs_B1_native=gained,lost_vs_B1_native=lost)
            comparisons[key][condition]=paired_effects(matrices)
    output=a.output/'results';output.mkdir(exist_ok=True)
    save_json(output/'edsr_asf_summary.json',dict(identity=a.trial_identity,recipe=RECIPE,scores=scores,comparisons=comparisons,
        timing=timing,transitions=transitions,no_test=True,no_training=True,
        caveats=['repeatedly used development; one detector seed; unadjusted multiple-endpoint source CIs',
                 'always-on restoration including Clean; no oracle corruption routing',
                 'shared bicubic synthetic degradation may favor pretrained EDSR',
                 'different-allocation absolute timings are descriptive; matched-ID transitions are not verified causes']))
    lines=['# EDSR x completed ASF: factorial inference study','',
        'Frozen completed B1 and completed ASF checkpoints; no training, retuning or TEST.',
        'B1 controls imported from completed job163012 after exact per-source and restored-byte checks.',
        '2262 paired Clean/LR2 sources and the same500-source stress cohort. Original native-LR2 size bins.','']
    for pair in PAIRS:
        key=f'{pair[0]:.2f}/{pair[1]:.2f}'
        lines += [f'## Fixed mask/score {key}'+(' (primary)' if pair==PAIRS[PRIMARY] else ' (historical)'),'',
            '| Condition | Pipeline | P | R | Hmean | FP | FN | Recall <4px | Median ms |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for condition in RECIPE['conditions']:
            for cell in CELLS:
                s=scores[key][condition][cell];gt,tp=s['bins_lr2_reference'][0]
                lines.append(f'| {condition} | {cell} | {s["precision"]:.2%} | {s["recall"]:.2%} | {s["hmean"]:.2%} | '
                    f'{s["fp"]} | {s["fn"]} | {tp/max(gt,1):.2%} | {timing[condition][cell]["median_total_ms"]:.1f} |')
        lines+=['','| Condition | Contrast | Delta H pp (95%CI) | Delta tiny recall pp (95%CI) |','|---|---|---:|---:|']
        for condition in ('clean','lr2'):
            for label,v in comparisons[key][condition].items():
                h,t=v['hmean_pp'],v['recall_lt4_pp']
                lines.append(f'| {condition} | {label} | {h["delta"]:+.3f} {h["ci95"]} | {t["delta"]:+.3f} {t["ci95"]} |')
    lines+=['','Interaction=(EDSR+ASF - ASF) - (EDSR+B1 - B1), computed on paired source resamples; gains are not added from different studies.',
        'JSON includes all size bins, P/R/FP intervals, source-paired contrasts and500-source Clean/LR2 anchors.',
        'This is exploratory development, one seed, multiple unadjusted endpoints; no final winner is automatically selected.',
        'B1 and ASF absolute timings come from different allocations and are not a controlled deployment latency comparison.','']
    write_text(output/'edsr_asf_report.md','\n'.join(lines))
    save_json(output/'complete.json',dict(identity=a.trial_identity,no_training=True,no_test=True,
        report_sha256=sha256(output/'edsr_asf_report.md'),summary_sha256=sha256(output/'edsr_asf_summary.json')))
