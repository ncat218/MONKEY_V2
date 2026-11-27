"""Fixed-window zoom+EDSR x B1/ASF x original frozen quality scorer.

Only Clean/native LR2. The four zoom arms share the original B1-selected windows,
crop restoration and replace fusion. Quality features come from the originating
global/crop P-map before coordinate conversion; raw scores still order NMS.
"""
import copy
from pathlib import Path
import time
import numpy as np

from lowres.common import read_json,sha256,identity
from rev4 import require
from rev4 import zoom_core as Z
from rev4.bundle_plan import ZOOM
from rev4.durable_io import save_json,write_text

MODELS=('B1','ASF')
ARMS=('F','Q')
CELLS=tuple(f'{m}/{mode}/{s}' for m in MODELS for mode in ('global','zoom_edsr') for s in ARMS)
CONTRASTS=(('B1/zoom_edsr/Q','B1/zoom_edsr/F'),('ASF/zoom_edsr/F','B1/zoom_edsr/F'),
    ('ASF/zoom_edsr/Q','ASF/zoom_edsr/F'),('ASF/zoom_edsr/Q','B1/zoom_edsr/Q'),
    ('ASF/zoom_edsr/Q','B1/global/F'),('B1/zoom_edsr/Q','B1/global/F'),
    ('ASF/zoom_edsr/F','B1/global/F'),('B1/zoom_edsr/F','B1/global/F'))


def predictions(pred,raw,features,model,ratio,source,window=-1):
    from mixab.boxquality_devA_janek import full_scores
    require(len(raw)==len(features)==len(pred['polygons']),'Feature/candidate order changed')
    require(all(len(f)==13 and np.isfinite(f).all() for f in features),'Invalid crop features')
    values=Z.make_preds(pred,raw,ratio,source,window)
    floor=[i for i,r in enumerate(raw) if r>=.20]
    q=full_scores(model,[features[i] for i in floor])
    # Preserve original raw/packed scores for NMS and official matching.
    return dict(F=[p for p in values if p['score']>=.6],
                Q=[values[i] for i,score in zip(floor,q) if score>=model['cut_full']])


def capture(native,probability,sample,restore,model,ratio,source,window=-1,origin=None,to_clean=None):
    from mixab.boxquality_janek import capture_with_features
    from rev4.decision_core import capture_candidates
    from rev4.quality_combo import same_candidates
    plain,raw,_=capture_candidates(native,probability,sample,restore,('fixed',.4,1.5,5,.01))
    start=time.perf_counter()
    pred,observed,features=capture_with_features(native,probability,sample,restore)
    require(same_candidates((plain,raw),(pred,observed)),'Crop feature extraction altered candidates')
    if origin is not None:pred=Z.shift_scale(pred,origin,to_clean)
    return predictions(pred,observed,features,model,ratio,source,window), (time.perf_counter()-start)*1000


def fused(global_preds,crops,rects,image_rect,params,arm):
    # Both pools are already filtered at their own frozen cutoffs. A zero cutoff
    # here must not discard a Q-retained box whose raw score is below .60.
    return Z.fuse(global_preds[arm],[p[arm] for p in crops],rects,image_rect,'replace',0.,params)


def assert_control(record,reference,old_zoom=None):
    for arm in ARMS:
        expected=reference['branches']['native']['arms'][arm]
        want=expected['counts']+sum(expected['bins'],[])
        require(record['results']['global/'+arm][:11]==want,'Global quality control differs from completed reference')
    if old_zoom is not None:
        require(record['windows']==old_zoom['windows']['guided'],'Frozen crop windows differ')
        require(record['results']['zoom_edsr/F']==old_zoom['results']['guided_edsr']['0.40/0.60'],
                'Z0 differs from completed B1 guided_edsr result')


def score(preds,row,policy,official):
    from rev4.zoom_study import score_preds
    counts,matcher=score_preds(preds,0.,row,policy)
    if official:matcher.assert_official(0.)
    return counts


def one(ctx,name,task,row,reference,detector,native,sr,device,model,policy,official=False):
    from rev4.stage_a import forward
    from rev4.zoom_study import load_task_image,forward_array
    from rev4.edsr import restore
    old=reference[task['sample_id']]
    require(old['image_sha256']==task['image_sha256'] and old['size']==task['size'],
            'Frozen window source bytes differ')
    start=time.perf_counter()
    _,prob,sample,_,scale,_,_=forward(detector,native,task,row)
    ratio=Z.ref_ratio(row['clean_size'])
    global_preds,feature_ms=capture(native,prob,sample,scale,model,ratio,'global')
    results={'global/'+arm:score(global_preds[arm],row,policy,official) for arm in ARMS}
    windows=old['windows']['guided'];image=load_task_image(task['path']);crops=[];rects=[]
    require(image.shape[:2]==tuple(reversed(task['size'])),'Decoded crop source size differs')
    for index,(x0,y0,x1,y1) in enumerate(windows):
        require(0<=x0<x1<=image.shape[1] and 0<=y0<y1<=image.shape[0],'Crop outside image')
        rgb=np.ascontiguousarray(image[y0:y1,x0:x1,::-1])
        restored=restore(sr,rgb,device)
        require(restored.shape==(2*(y1-y0),2*(x1-x0),3),'EDSR crop shape changed')
        _,pmap,crop_sample=forward_array(detector,np.ascontiguousarray(restored[:,:,::-1]))
        current,ms=capture(native,pmap,crop_sample,np.array([.5,.5]),model,ratio,'zoom',index,
                           (x0,y0),scale)
        feature_ms+=ms;crops.append(current);rects.append(Z.rect_to_clean((x0,y0,x1,y1),scale))
    params=dict(ZOOM);params['margin_clean']=params['margin_task_px']*float(max(scale))
    image_rect=(0.,0.,float(row['clean_size'][0]),float(row['clean_size'][1]))
    for arm in ARMS:
        combined=fused(global_preds,crops,rects,image_rect,params,arm)
        results['zoom_edsr/'+arm]=score(combined,row,policy,official)
    result=dict(sample_id=task['sample_id'],source_image_id=task['source_image_id'],condition=task['condition'],
        detector=name,input_sha256=task['image_sha256'],identity_sha256=identity(ctx.trial_identity),
        results=results,windows=windows,crop_count=len(windows),official_checked=official,
        diagnostic_ms=(time.perf_counter()-start)*1000,feature_ms=feature_ms,
        timing_scope='shared F/Q passes plus parity checks and evaluation; not deployment latency')
    quality=read_json(ctx.quality_reference/'records'/name/task['condition']/(task['source_image_id']+'.json'))
    require(quality['identity_sha256']==ctx.quality_identity_sha,'Wrong global quality reference')
    assert_control(result,quality,old if name=='B1' else None)
    return result


def validate(ctx,name,task,value,reference):
    require(value['identity_sha256']==identity(ctx.trial_identity) and value['sample_id']==task['sample_id']
        and value['detector']==name and value['input_sha256']==task['image_sha256'],'Zoom cache identity differs')
    q=read_json(ctx.quality_reference/'records'/name/task['condition']/(task['source_image_id']+'.json'))
    assert_control(value,q,reference[task['sample_id']] if name=='B1' else None)
    require(value['windows']==reference[task['sample_id']]['windows']['guided'],'Cache windows changed')
    require(set(value['results'])=={'global/F','global/Q','zoom_edsr/F','zoom_edsr/Q'},'Missing zoom arm')


def run(ctx,probe_only=False):
    import gc
    import torch
    from rev4.calibration import validation_cohort
    from rev4.edsr import EDSR
    from rev4.quality_combo import frozen_model
    from rev4.sr_recheck import completed_b1_detector
    from rev4.sr_asf import completed_asf_detector
    rows,tasks=validation_cohort(ctx,ctx.calibration_prepared)
    require(len(rows)==2262 and len(tasks)==4524,'Zoom main cohort changed')
    old_path=ctx.research_reference/'zoom/B1/records.json'
    marker=read_json(ctx.research_reference/'zoom/B1/complete.json')
    require(sha256(old_path)==marker['records_sha256'],'Frozen B1 zoom record changed')
    old=read_json(old_path);reference={v['sample_id']:v for v in old['tasks']}
    model=frozen_model(ctx.repo);policy=read_json(ctx.evaluation_protocol)
    sr=EDSR().float().eval().requires_grad_(False)
    sr.load_state_dict(torch.load(ctx.edsr_weight,map_location='cpu',weights_only=True),strict=True)
    device=torch.device('cuda:0');sr.to(device)
    root=ctx.output/'zoom_quality';probe=root/'preflight.json'
    receipt=read_json(probe) if probe.exists() else dict(identity=ctx.trial_identity,models={})
    require(receipt['identity']==ctx.trial_identity,'Zoom probe identity changed')
    # Probe both models, both conditions before the full pass. Frozen windows use no GT.
    for name,loader in [('B1',completed_b1_detector),('ASF',completed_asf_detector)]:
        if name in receipt['models']:continue
        detector,native=loader(ctx);checked=[]
        for condition in ('clean','lr2'):
            chosen=sorted([t for t in tasks if t['condition']==condition],key=lambda t:t['source_image_id'])[:2]
            for task in chosen:
                value=one(ctx,name,task,rows[task['source_image_id']],reference,detector,native,sr,device,model,policy,True)
                save_json(root/'records'/name/task['condition']/(task['source_image_id']+'.json'),value)
                checked.append(task['sample_id'])
        receipt['models'][name]=checked;save_json(probe,receipt)
        del detector,native;gc.collect();torch.cuda.empty_cache()
    receipt.update(status='PASS',no_training=True,no_test=True);save_json(probe,receipt)
    print('PASS zoom quality: both models, fixed windows, crop coordinates, global controls, Z0 and official matching',flush=True)
    if probe_only:return
    for name,loader in [('B1',completed_b1_detector),('ASF',completed_asf_detector)]:
        detector,native=loader(ctx)
        for number,task in enumerate(tasks,1):
            sid=task['source_image_id'];row=rows[sid]
            require(sha256(ctx.clean_root/row['clean_relpath'])==row['clean_sha256'],'Clean bytes changed')
            target=root/'records'/name/task['condition']/(sid+'.json')
            if target.exists():validate(ctx,name,task,read_json(target),reference)
            else:save_json(target,one(ctx,name,task,row,reference,detector,native,sr,device,model,policy))
            if number%50==0 or number==len(tasks):
                save_json(root/'progress.json',dict(detector=name,completed_tasks=number,total_tasks=len(tasks)))
                print('Zoom quality',name,number,'/',len(tasks),flush=True)
        del detector,native;gc.collect();torch.cuda.empty_cache()
    report(ctx,tasks)


def report(ctx,tasks):
    from rev4.sr_recheck import metric_vector,METRICS
    scores={};contrasts={};timing={}
    for condition in ('clean','lr2'):
        group=sorted([t for t in tasks if t['condition']==condition],key=lambda t:t['source_image_id'])
        matrices={k:[] for k in CELLS}
        for name in MODELS:
            values=[read_json(ctx.output/'zoom_quality/records'/name/condition/(t['source_image_id']+'.json')) for t in group]
            for mode in ('global','zoom_edsr'):
                for arm in ARMS:
                    matrices[f'{name}/{mode}/{arm}']=[v['results'][mode+'/'+arm][:11] for v in values]
            timing.setdefault(condition,{})[name]=dict(median_diagnostic_ms=float(np.median([v['diagnostic_ms'] for v in values])),
                mean_crops=float(np.mean([v['crop_count'] for v in values])),scope='shared diagnostic; not deployment')
        # Add the already completed whole-image winning recipe as an actual paired control.
        matrices['whole_ASF_EDSR_Q']=[read_json(ctx.quality_reference/'records/ASF'/condition/(t['source_image_id']+'.json'))
            ['branches']['edsr_x2']['arms']['Q'] for t in group]
        matrices['whole_ASF_EDSR_Q']=[v['counts']+sum(v['bins'],[]) for v in matrices['whole_ASF_EDSR_Q']]
        pairs=CONTRASTS+(('ASF/zoom_edsr/Q','whole_ASF_EDSR_Q'),('B1/zoom_edsr/Q','whole_ASF_EDSR_Q'))
        arrays={k:np.asarray(v,dtype=np.int64) for k,v in matrices.items()}
        points={k:metric_vector(v.sum(0)) for k,v in arrays.items()}
        rng=np.random.default_rng(20261107);samples={k:[] for k in arrays}
        for start in range(0,2000,50):
            take=rng.integers(0,len(group),size=(50,len(group)))
            for k,v in arrays.items():samples[k].append(metric_vector(v[take].sum(1)))
        samples={k:np.concatenate(v) for k,v in samples.items()};contrasts[condition]={};scores[condition]={}
        for key,array in arrays.items():
            t=array.sum(0);scores[condition][key]=dict(tp=int(t[0]),gt=int(t[1]),pred=int(t[2]),fp=int(t[2]-t[0]),
                fn=int(t[1]-t[0]),hmean=float(points[key][0]),precision=float(points[key][1]),recall=float(points[key][2]),
                tiny_recall=float(points[key][4]),bins=t[3:].reshape(4,2).tolist())
        for new,old in pairs:
            delta=points[new]-points[old];ci=np.quantile(samples[new]-samples[old],[.025,.975],axis=0)
            contrasts[condition][new+' minus '+old]={k:dict(delta=float(delta[i]),ci95=ci[:,i].tolist()) for i,k in enumerate(METRICS)}
    root=ctx.output/'zoom_quality/results'
    save_json(root/'summary.json',dict(identity=ctx.trial_identity,scores=scores,contrasts=contrasts,timing=timing,
        no_training=True,no_test=True,caveats=['reused development, one seed, multiple endpoints',
        'Q frozen from whole-image B1 and transferred to crops; no fitting or cutoff selection',
        'same B1-selected windows for both detectors; raw order preserved in NMS']))
    lines=['# Fixed-window zoom + EDSR x ASF x frozen Q','',
        'Clean/LR2:2262 paired development sources. Four zoom arms, same original B1 windows and replace fusion. No TEST.','',
        '| Condition | Pipeline | Hmean | P | R | Tiny recall | FP |','|---|---|---:|---:|---:|---:|---:|']
    for condition in ('clean','lr2'):
        for cell,s in scores[condition].items():lines.append(f'| {condition} | {cell} | {s["hmean"]:.2f}% | {s["precision"]:.2f}% | '
            f'{s["recall"]:.2f}% | {s["tiny_recall"]:.2f}% | {s["fp"]} |')
    lines+=['','| Condition | Contrast | Delta H pp (95% CI) | Delta tiny pp (95% CI) |','|---|---|---:|---:|']
    for condition,values in contrasts.items():
        for label,v in values.items():
            h,t=v['hmean_pp'],v['recall_lt4_pp']
            lines.append(f'| {condition} | {label} | {h["delta"]:+.3f} {h["ci95"]} | {t["delta"]:+.3f} {t["ci95"]} |')
    lines+=['','Timing in JSON includes shared extraction, parity checks and scoring; it is not deployable end-to-end latency.',
        'Source CIs exclude training-seed variation and selection bias; no automatic winner is selected.','']
    write_text(root/'report.md','\n'.join(lines))
    save_json(root/'complete.json',dict(identity=ctx.trial_identity,no_test=True,no_training=True,
        report_sha256=sha256(root/'report.md'),summary_sha256=sha256(root/'summary.json')))
