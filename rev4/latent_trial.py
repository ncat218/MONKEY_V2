"""Equal-budget head/plain/gated pilot, with baseline and frozen-state gates."""
import copy
from pathlib import Path
import time

import numpy as np

from lowres.common import read_json,save_json,sha256
from lowres.evaluate_shared import evaluate_image,rates,bootstrap_paired
from rev4 import require,b1_training,calibration
from rev4.latent_plan import ARMS,PAIRS,RECIPE
from rev4.stage_a import detector_for,forward


def configuration(a,arm):
    cfg=b1_training.configuration(a,a.calibration_prepared,a.output/arm/'weights',None)
    cfg.load_from=None
    cfg.resume=False
    cfg.model.type='LatentPilotDBNet'
    cfg.model.adapter_kind=arm
    cfg.model.backbone['freeze']=True
    cfg.custom_imports['imports']+=['rev4.latent_plugins']
    cfg.custom_hooks=[dict(type='LatentAuditHook')]
    total=RECIPE['optimizer_updates']*RECIPE['accumulation']
    warmup=RECIPE['warmup_updates']*RECIPE['accumulation']
    cfg.train_cfg.max_iters=total
    cfg.default_hooks.checkpoint.update(interval=total,max_keep_ckpts=1)
    cfg.optim_wrapper.optimizer.lr=RECIPE['learning_rate']
    cfg.param_scheduler=[dict(type='LinearLR',start_factor=.1,begin=0,end=warmup,by_epoch=False),
        dict(type='PolyLR',power=.9,eta_min=RECIPE['learning_rate']*.01,begin=warmup,end=total,by_epoch=False)]
    return cfg


def runner_for(a,arm):
    import torch
    from mmengine.runner import Runner
    from mmengine.utils import import_modules_from_strings
    from mmocr.utils import register_all_modules
    from rev4.latent_modules import load_base_state

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    register_all_modules(init_default_scope=True)
    cfg=configuration(a,arm)
    import_modules_from_strings(**cfg.custom_imports)
    runner=Runner.from_cfg(cfg)
    runner.model.init_weights()
    runner.model.float()
    if arm!='head':torch.nn.init.zeros_(runner.model.feature_adapter.project_out.weight)
    checkpoint=torch.load(a.b1_checkpoint,map_location='cpu',weights_only=False)
    load_base_state(runner.model,checkpoint['state_dict'])
    from rev4.latent_modules import frozen_digest
    object.__setattr__(runner.model,'_initial_frozen_digest',frozen_digest(runner.model))
    object.__setattr__(runner.model,'_initial_head_state',
        {k:v.detach().cpu().clone() for k,v in runner.model.det_head.state_dict().items()})
    require(all(p.dtype==torch.float32 for p in runner.model.parameters() if p.is_floating_point()),'Expected FP32 parameters')
    return runner


def probe(a):
    import torch
    import hashlib
    from rev4.latent_modules import frozen_digest

    require(b1_training.environment_identity(a,a.calibration_prepared)==a.b1_identity,'Historical runtime changed')
    rows,tasks=calibration.validation_cohort(a,a.calibration_prepared)
    options=copy.copy(a);options.checkpoint=a.b1_checkpoint
    detector,pp,_,_=detector_for(options)
    pp.mask_thr,pp.min_text_score=PAIRS[1]
    sources=set(sorted(rows)[:2])
    baseline=[]
    for task in tasks:
        if task['source_image_id'] not in sources:continue
        processed,pmap,*_=forward(detector,pp,task,rows[task['source_image_id']])
        baseline.append((processed,pmap.detach().cpu(),task['sample_id']))
    require(len(baseline)==4,'Missing baseline probe images')
    del detector
    probes={}
    common_draws=None
    for arm in ARMS:
        runner=runner_for(a,arm)
        model=runner.model
        model.eval()
        errors=[]
        for processed,target,sample_id in baseline:
            with torch.inference_mode():actual=model(**processed,mode='tensor')
            error=float((actual.detach().cpu()-target).abs().max())
            require(error<=1e-6,f'{arm} identity initialization failed: {sample_id}, {error}')
            errors.append(error)
        model.train()
        frozen_before=frozen_digest(model)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=RECIPE['learning_rate'])
        loader=iter(runner.train_dataloader)
        draws=[];output_grad=False;body_grad=False
        try:
            for step in range(4):
                batch=next(loader)
                conditions=[s.get('condition') for s in batch['data_samples']]
                require(conditions==['clean','lr2'],'Unbalanced probe TRAIN draw')
                digest=hashlib.sha256()
                for image in batch['inputs']:digest.update(image.cpu().contiguous().numpy().tobytes())
                draws.append(dict(sample_ids=[s.get('sample_id') for s in batch['data_samples']],augmented_input_sha256=digest.hexdigest()))
                processed=model.data_preprocessor(batch,training=True)
                losses=model(**processed,mode='loss')
                parsed,_=model.parse_losses(losses)
                require(torch.isfinite(parsed),'Nonfinite probe detection loss')
                optimizer.zero_grad(set_to_none=True)
                parsed.backward()
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                require(gradients and all(torch.isfinite(g).all() for g in gradients),'Invalid probe gradients')
                if arm!='head':
                    adapter=model.feature_adapter
                    output_grad|=adapter.project_out.weight.grad is not None and bool(adapter.project_out.weight.grad.abs().sum()>0)
                    body_grad|=adapter.project_in.weight.grad is not None and bool(adapter.project_in.weight.grad.abs().sum()>0)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],RECIPE['clip_grad'],error_if_nonfinite=True)
                optimizer.step()
        finally:
            if hasattr(loader,'_shutdown_workers'):loader._shutdown_workers()
        require(frozen_digest(model)==frozen_before,'Frozen backbone/FPNC state changed during probe')
        require(all(p.grad is None for c in (model.backbone,model.neck) for p in c.parameters()),'Frozen parameter gradient detected')
        require(arm=='head' or (output_grad and body_grad),'Adapter output/body failed to receive gradients')
        if common_draws is None:common_draws=draws
        require(draws==common_draws,'TRAIN source/augmentation stream differs between arms')
        probes[arm]=dict(max_identity_error=max(errors),frozen_state_unchanged=True,draws=draws,
            adapter_parameters=sum(p.numel() for p in model.feature_adapter.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            output_gradient=output_grad,body_gradient=body_grad)
        del runner,model,optimizer
        torch.cuda.empty_cache()
    save_json(a.output/'preflight.json',dict(status='PASS',identity=a.trial_identity,probes=probes))
    print('PASS: identity, real TRAIN gradients, identical draws/augmentations and frozen state',flush=True)


def train(a,arm):
    import torch
    from rev4.latent_plugins import LatentAuditHook
    receipt=a.output/arm/'trained.json'
    if receipt.is_file():
        prior=read_json(receipt)
        require(prior['identity']==a.trial_identity and sha256(prior['checkpoint'])==prior['checkpoint_sha256'],'Trained checkpoint changed')
        return
    started=a.output/arm/'started.json'
    require(not started.exists(),'Partial training found; review before spending another training budget')
    save_json(started,dict(identity=a.trial_identity,arm=arm))
    runner=runner_for(a,arm)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    runner.train()
    require(runner.iter==2*RECIPE['optimizer_updates'],'Latent training ended early')
    audit=next(h for h in runner.hooks if isinstance(h,LatentAuditHook))
    require(audit.batches==runner.iter,'Missing training audit iterations')
    checkpoint=a.output/arm/'weights'/f'iter_{runner.iter}.pth'
    save_json(receipt,dict(identity=a.trial_identity,arm=arm,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        optimizer_updates=RECIPE['optimizer_updates'],micro_iterations=runner.iter,draws=audit.draws,
        frozen_digest=audit.start_digest,seconds=time.perf_counter()-start,
        peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2))


def inference_configuration(a,arm):
    from mmengine.config import Config
    cfg=Config.fromfile(str(a.model_config))
    imports=list(cfg.get('custom_imports',{}).get('imports',[]))
    cfg.custom_imports=dict(imports=imports+['rev4.latent_plugins'],allow_failed_imports=False)
    cfg.model.type='LatentPilotDBNet';cfg.model.adapter_kind=arm
    cfg.model.backbone['freeze']=True
    return cfg


def evaluate(a,arm):
    from shapely.geometry import shape
    from lowres.inference import unpack
    trained=read_json(a.output/arm/'trained.json')
    require(trained['identity']==a.trial_identity and sha256(trained['checkpoint'])==trained['checkpoint_sha256'],'Checkpoint mismatch')
    cfg=inference_configuration(a,arm)
    config_path=a.output/arm/'inference_config.py';cfg.dump(str(config_path))
    options=copy.copy(a);options.model_config=config_path;options.checkpoint=Path(trained['checkpoint'])
    detector,native,_,_=detector_for(options)
    processors=[]
    for pair in PAIRS:
        processor=copy.copy(native);processor.mask_thr,processor.min_text_score=pair
        processors.append(processor)
    rows,tasks=calibration.validation_cohort(a,a.calibration_prepared)
    policy=read_json(a.evaluation_protocol)
    target=a.output/arm/'validation.json'
    record=read_json(target) if target.is_file() else dict(identity=a.trial_identity,arm=arm,tasks=[])
    require(record['identity']==a.trial_identity and [t['sample_id'] for t in record['tasks']]==[t['sample_id'] for t in tasks[:len(record['tasks'])]],'Evaluation cache changed')
    for number,task in enumerate(tasks[len(record['tasks']):],len(record['tasks'])+1):
        row=rows[task['source_image_id']]
        _,pmap,sample,_,restore,_,_=forward(detector,native,task,row)
        scored=[];bins=[]
        for processor in processors:
            prediction=unpack(processor(pmap,[copy.deepcopy(sample)])[0],restore)
            result=evaluate_image(row['gt_clean'],row['roi_clean'],prediction,policy)
            matched=set(map(str,result['matched_ann_ids']))
            group=[[0,0] for _ in range(4)]
            for gt in row['gt_clean']['valid']:
                height=(shape(gt['geometry']).bounds[3]-shape(gt['geometry']).bounds[1])*(max(1,row['clean_size'][1]//2)/row['clean_size'][1])
                i=sum(height>=cut for cut in (4,8,16));group[i][0]+=1;group[i][1]+=int(str(gt['ann_id']) in matched)
            scored.append(result['counts']);bins.append(group)
        record['tasks'].append(dict(sample_id=task['sample_id'],source_image_id=task['source_image_id'],condition=task['condition'],counts=scored,bins=bins))
        if number%100==0 or number==len(tasks):
            save_json(target,record);print(f'Latent {arm} validation {number}/{len(tasks)}',flush=True)
    del detector
    latency(a,arm,config_path,Path(trained['checkpoint']),rows,tasks)


def latency(a,arm,config_path,checkpoint,rows,tasks):
    """Within-phase B1/arm timing: same eight inputs, GPU model + CPU fixed decode."""
    import torch
    from lowres.inference import unpack
    target=a.output/arm/'latency.json'
    if target.is_file():
        require(read_json(target)['identity']==a.trial_identity,'Latency cache changed')
        return
    selected_sources=set(sorted(rows)[:4])
    selected=[t for t in tasks if t['source_image_id'] in selected_sources]
    result={}
    for label,config,weight in [('B1',a.model_config,a.b1_checkpoint),(arm,config_path,checkpoint)]:
        options=copy.copy(a);options.model_config=config;options.checkpoint=weight
        detector,native,_,_=detector_for(options)
        processor=copy.copy(native);processor.mask_thr,processor.min_text_score=PAIRS[1]
        measurements=[]
        torch.cuda.reset_peak_memory_stats()
        for task in selected:
            processed,_,sample,_,restore,_,_=forward(detector,native,task,rows[task['source_image_id']])
            with torch.inference_mode():
                for _ in range(2):detector.model(**processed,mode='tensor')
                torch.cuda.synchronize()
                values=[]
                for _ in range(5):
                    start=time.perf_counter()
                    pmap=detector.model(**processed,mode='tensor')
                    unpack(processor(pmap,[copy.deepcopy(sample)])[0],restore)
                    torch.cuda.synchronize()
                    values.append(1000*(time.perf_counter()-start))
            measurements.append(dict(sample_id=task['sample_id'],condition=task['condition'],median_ms=float(np.median(values))))
        result[label]=dict(measurements=measurements,
            median_ms={c:float(np.median([v['median_ms'] for v in measurements if v['condition']==c])) for c in ('clean','lr2')},
            peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)
        del detector
        torch.cuda.empty_cache()
    save_json(target,dict(identity=a.trial_identity,results=result,
        scope='eight fixed images; two warmups, five repeats; model and CPU fixed quad decode; excludes file IO/preprocess/evaluation',
        limitation='short within-phase screen, not isolated deployment benchmark'))


def report(a):
    rows,tasks=calibration.validation_cohort(a,a.calibration_prepared)
    sources=sorted(rows);position={s:i for i,s in enumerate(sources)}
    matrices={arm:np.zeros((len(rows),2,len(PAIRS),4),dtype=np.int64) for arm in ARMS}
    training={arm:read_json(a.output/arm/'trained.json') for arm in ARMS}
    require(all(v['draws']==training['head']['draws'] for v in training.values()),'Training streams differ')
    bins={}
    for arm in ARMS:
        record=read_json(a.output/arm/'validation.json')
        require(len(record['tasks'])==len(tasks) and record['identity']==a.trial_identity,'Incomplete arm evaluation')
        for task in record['tasks']:
            matrices[arm][position[task['source_image_id']],int(task['condition']=='lr2')]=task['counts']
        bins[arm]={c:np.asarray([t['bins'][1] for t in record['tasks'] if t['condition']==c]).sum(axis=0).tolist() for c in ('clean','lr2')}
    # Existing fixed B1/B-Clean counts are the strong external baselines.
    for name in ('B-Clean','B1'):
        matrix=np.zeros((len(rows),2,1,4),dtype=np.int64)
        for task in tasks:
            rec=read_json(a.fixed_run/'models'/name/task['condition']/f'{task["source_image_id"]}.json')
            result=rec['results']['0.40/0.60']
            matrix[position[task['source_image_id']],int(task['condition']=='lr2'),0]=result['counts']
        matrices[name]=matrix
    scores={}
    for name,matrix in matrices.items():
        scores[name]={}
        for j,c in enumerate(('clean','lr2')):
            totals=matrix[:,j,-1,:].sum(axis=0).tolist();tp,gt,pred,_=totals
            scores[name][c]=dict(**rates(totals),tp=tp,fp=pred-tp,fn=gt-tp)
    comparisons={}
    for newer,older in [('plain','head'),('gated','plain'),('gated','head'),('gated','B1'),('plain','B1')]:
        comparisons[newer+'_minus_'+older]={c:bootstrap_paired(np.concatenate([matrices[older][:,j,-1,:],matrices[newer][:,j,-1,:]],axis=1),2000,20261107) for j,c in enumerate(('clean','lr2'))}
    summary=dict(identity=a.trial_identity,scores=scores,comparisons=comparisons,height_bins=bins,training=training,
        latency={arm:read_json(a.output/arm/'latency.json') for arm in ARMS},
        precision='fp32',primary_pair=list(PAIRS[1]),no_test=True,seed=42,checkpoint_selection='final only',
        limitations=['1000-update screening','one training seed','development already used','short within-phase timing only'])
    output=a.output/'results';output.mkdir(parents=True,exist_ok=True)
    save_json(output/'latent_summary.json',summary)
    lines=['# Latent refinement pilot','',
        'Same B1 init, 1000 optimizer updates per arm; frozen backbone/FPNC including buffers. DB detection loss only.',
        'Primary fixed mask=.40 / score=.60; no threshold or checkpoint search. Development data only; no TEST.','',
        '| Model | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP |','|---|---:|---:|---:|---:|---:|']
    for name,value in scores.items():
        c,d=value['clean'],value['lr2']
        lines.append(f'| {name} | {c["hmean"]:.2%} | {d["hmean"]:.2%} | {d["precision"]:.2%} | {d["recall"]:.2%} | {d["fp"]} |')
    lines+=['','## Paired source CI (2000 draws, seed 20261107)','']
    for label,conditions in comparisons.items():
        newer,older=label.split('_minus_')
        for c,value in conditions.items():
            delta=100*(scores[newer][c]['hmean']-scores[older][c]['hmean'])
            lines.append(f'- {label}, {c}: {delta:+.3f} pp; 95% CI {value["delta_pp_95ci"]["hmean"]}.')
    lines+=['','One seed and short pilot; source CIs do not quantify training-seed or selection uncertainty. '
             'No automatic winner or final TEST decision. See JSON for four GT height bins and training cost.','']
    (output/'latent_report.md').write_text('\n'.join(lines),encoding='utf-8')
    save_json(output/'complete.json',dict(no_test=True,report_sha256=sha256(output/'latent_report.md'),identity=a.trial_identity))
    print('Latent pilot complete:',output/'latent_report.md',flush=True)
