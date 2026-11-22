"""Equal-budget trainable-neck study; the pilot gate is inside the full run."""
import copy
from pathlib import Path
import random
import time
import numpy as np

from lowres.common import read_json,sha256
from rev4 import require,b1_training,calibration
from rev4.durable_io import save_json
from rev4.fusion_plan import ARMS,RECIPE


def configuration(a,arm,probe=False):
    cfg=b1_training.configuration(a,a.calibration_prepared,a.output/('probe' if probe else 'arms')/arm/'weights',None)
    cfg.load_from=None;cfg.resume=False
    cfg.model.type='FusionStudyDBNet';cfg.model.backbone['freeze']=True
    cfg.model.neck.type='StudyFPNC';cfg.model.neck.mode=arm
    cfg.model.neck.asf_cfg=dict(attention_type='ScaleChannelSpatial') if arm=='asf' else None
    cfg.custom_imports['imports']+=['rev4.fusion_plugins']
    cfg.custom_hooks=[dict(type='FusionAuditHook',pilot_updates=2 if probe else RECIPE['pilot_updates'],accumulation=2)]
    cfg.train_cfg.max_iters=4 if probe else 20000
    cfg.default_hooks.checkpoint.update(interval=4 if probe else 5000,max_keep_ckpts=1 if probe else 4)
    cfg.param_scheduler=[dict(type='LinearLR',start_factor=.1,begin=0,end=200,by_epoch=False),
                         dict(type='PolyLR',power=.9,eta_min=2e-7,begin=200,end=20000,by_epoch=False)]
    return cfg


def runner_for(a,arm,probe=False):
    import torch
    from mmengine.runner import Runner,set_random_seed
    from mmengine.utils import import_modules_from_strings
    from mmocr.utils import register_all_modules
    from rev4.fusion_modules import load_shared,state_digest
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    register_all_modules(init_default_scope=True)
    cfg=configuration(a,arm,probe)
    import_modules_from_strings(**cfg.custom_imports)
    runner=Runner.from_cfg(cfg);model=runner.model
    model.init_weights();model.float()
    load_shared(model,torch.load(a.b1_checkpoint,map_location='cpu',weights_only=False)['state_dict'])
    if arm=='dysample':
        for module in list(model.neck.top_sample)+list(model.neck.output_sample):
            torch.nn.init.zeros_(module.offset.weight);torch.nn.init.zeros_(module.offset.bias)
    object.__setattr__(model,'_backbone_digest',state_digest(model.backbone))
    object.__setattr__(model,'_initial_trainable',{k:v.detach().cpu().clone() for k,v in model.state_dict().items()
                                                 if k.startswith(('neck.','det_head.'))})
    require(all(p.dtype==torch.float32 for p in model.parameters()),'Require FP32 parameters')
    # New-module construction must not alter the shared data/augmentation RNG stream.
    set_random_seed(42,deterministic=False)
    return runner


def probe(a):
    import torch
    from rev4.stage_a import detector_for,forward
    from rev4.fusion_modules import state_digest
    from rev4.fusion_plugins import FusionAuditHook
    require(b1_training.environment_identity(a,a.calibration_prepared)==a.b1_identity,'Historical runtime changed')
    rows,tasks=calibration.validation_cohort(a,a.calibration_prepared)
    sources=set(sorted(rows)[:2]);options=copy.copy(a);options.checkpoint=a.b1_checkpoint
    detector,pp,_,_=detector_for(options)
    samples=[]
    for task in tasks:
        if task['source_image_id'] in sources:
            processed,pmap,*_=forward(detector,pp,task,rows[task['source_image_id']])
            samples.append((processed,pmap.cpu()))
    require(len(samples)==4,'Probe sample coverage changed')
    del detector
    baselines=[];streams=[];records={}
    for arm in ARMS:
        runner=runner_for(a,arm,True);runner.model.eval();errors=[];maps=[]
        for processed,b1map in samples:
            with torch.inference_mode():pmap=runner.model(**processed,mode='tensor').cpu()
            require(torch.isfinite(pmap).all(),'Nonfinite initial P-map')
            errors.append(float((pmap-b1map).abs().max()));maps.append(pmap)
        if arm=='nearest':require(max(errors)<=1e-6,'Nearest initialization failed to reproduce B1')
        if arm=='bilinear':baselines=maps
        if arm=='dysample':require(max(float((v-w).abs().max()) for v,w in zip(maps,baselines))<=1e-3,'Zero-offset DySample differs from bilinear beyond numeric tolerance')
        runner.train()
        audit=next(h for h in runner.hooks if isinstance(h,FusionAuditHook))
        require(audit.batches==4 and state_digest(runner.model.backbone)==runner.model._backbone_digest,'Probe audit failed')
        streams.append(audit.stream)
        records[arm]=dict(max_pmap_difference_from_B1=max(errors),stream=audit.stream,module_updates=audit.module_updates,
                          pilot_checkpoint_roundtrip=True,backbone_unchanged=True)
        del runner,audit;torch.cuda.empty_cache()
    require(all(v==streams[0] for v in streams),'Probe TRAIN/augmentation streams differ')
    save_json(a.output/'preflight.json',dict(status='PASS',identity=a.trial_identity,arms=records))
    print('PASS all fusion probes; full job may proceed',flush=True)


def train(a,arm):
    import torch
    from rev4.fusion_plugins import FusionAuditHook
    root=a.output/'arms'/arm;done=root/'trained.json'
    if done.exists():
        r=read_json(done)
        require(r['identity']==a.trial_identity and sha256(r['checkpoint'])==r['checkpoint_sha256'],'Trained weights changed')
        return
    require(not (root/'started.json').exists(),'Partial full training exists; audit before any restart')
    save_json(root/'started.json',dict(identity=a.trial_identity,arm=arm))
    runner=runner_for(a,arm);start=time.perf_counter();runner.train()
    audit=next(h for h in runner.hooks if isinstance(h,FusionAuditHook))
    require(runner.iter==audit.batches==20000,'Incomplete full training')
    require(read_json(root/'pilot_gate.json')['status']=='PASS','In-run pilot never passed')
    path=root/'weights/iter_20000.pth'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    steps=[int(v['step']) for v in checkpoint['optimizer']['state'].values() if 'step' in v]
    require(min(steps)==max(steps)==10000 and checkpoint['meta']['iter']==20000,'Final optimizer/iteration count differs')
    save_json(done,dict(identity=a.trial_identity,arm=arm,checkpoint=str(path),checkpoint_sha256=sha256(path),
                       optimizer_updates=10000,micro_iterations=20000,seconds=time.perf_counter()-start,
                       stream=audit.stream,module_updates=audit.module_updates,peak_gpu_mib=audit.peak_gpu_mib))
    if arm!='nearest':require(read_json(a.output/'arms/nearest/trained.json')['stream']==audit.stream,'Full TRAIN/augmentation stream differs from control')
    print('Full training completed:',arm,flush=True)
