"""FPNC variants based on OpenMMLab Apache-2.0 FPNC; upstream helpers unchanged."""
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from mmengine.hooks import Hook
from mmocr.models.textdet.detectors import DBNet
from mmocr.models.textdet.necks import FPNC
from mmocr.registry import MODELS,HOOKS

from rev4 import require
from rev4.b1_plugins import Rev4B1AuditHook
from rev4.durable_io import save_json
from rev4.fusion_modules import sampler,state_digest


@MODELS.register_module()
class StudyFPNC(FPNC):
    def __init__(self,mode='nearest',**kwargs):
        require(mode in ('nearest','asf','bilinear','dysample'),'Unknown fusion mode')
        self.mode=mode
        super().__init__(**kwargs)
        require((self.asf_cfg is not None)==(mode=='asf'),'Inconsistent ASF configuration')
        if mode=='dysample':
            self.top_sample=torch.nn.ModuleList([sampler(self.lateral_channels,2) for _ in range(3)])
            self.output_sample=torch.nn.ModuleList([sampler(self.out_channels,2**i) for i in range(1,4)])

    def forward(self,inputs):
        if self.mode in ('nearest','asf'):return super().forward(inputs)
        levels=[conv(x) for conv,x in zip(self.lateral_convs,inputs)]
        for i in range(3,0,-1):
            target=levels[i-1].shape[-2:]
            up=self.top_sample[i-1](levels[i]) if self.mode=='dysample' else F.interpolate(levels[i],size=target,mode='bilinear',align_corners=False)
            require(up.shape[-2:]==target,'Backbone grids are not exact dyadic grids')
            levels[i-1]=levels[i-1]+up
        outputs=[conv(x) for conv,x in zip(self.smooth_convs,levels)]
        aligned=[outputs[0]]
        for i in range(1,4):
            up=self.output_sample[i-1](outputs[i]) if self.mode=='dysample' else F.interpolate(outputs[i],size=outputs[0].shape[-2:],mode='bilinear',align_corners=False)
            require(up.shape[-2:]==outputs[0].shape[-2:],'Output grids differ')
            aligned.append(up)
        value=torch.cat(aligned,dim=1)
        return self.out_conv(value) if self.conv_after_concat else value


@MODELS.register_module()
class FusionStudyDBNet(DBNet):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_feat(self,inputs):
        with torch.no_grad():features=self.backbone(inputs)
        return self.neck(features)


@HOOKS.register_module()
class FusionAuditHook(Rev4B1AuditHook):
    def __init__(self,pilot_updates=1000,accumulation=2):
        self.pilot_iter=pilot_updates*accumulation
        self.accumulation=accumulation
        self.batches=0;self.peak_gpu_mib=0.

    def before_train(self,runner):
        model=runner.model
        require(state_digest(model.backbone)==model._backbone_digest,'Backbone initialization changed')
        require(all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in model._initial_trainable.items()),'Runner changed initial neck/head weights')
        self.source=hashlib.sha256();self.augmentation=hashlib.sha256()
        self.initial=model._initial_trainable

    def before_train_iter(self,runner,batch_idx,data_batch=None):
        super().before_train_iter(runner,batch_idx,data_batch)
        model=runner.model
        require(not model.backbone.training and all(not p.requires_grad and p.grad is None for p in model.backbone.parameters()),'Backbone unfrozen')
        for sample in data_batch['data_samples']:self.source.update((sample.get('sample_id')+'\n').encode())
        for tensor in data_batch['inputs']:
            self.augmentation.update(str(tuple(tensor.shape)).encode())
            self.augmentation.update(tensor.cpu().contiguous().numpy().tobytes())

    def deltas(self,runner):
        sums={'neck':0.,'det_head':0.,'new_modules':0.}
        for name,p in runner.model.named_parameters():
            if name not in self.initial:continue
            squared=float((p.detach().cpu()-self.initial[name]).double().square().sum())
            sums[name.split('.')[0]]+=squared
            if name.startswith(('neck.asf_','neck.top_sample.','neck.output_sample.')):sums['new_modules']+=squared
        return {k:v**.5 for k,v in sums.items()}

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        super().after_train_iter(runner,batch_idx,data_batch,outputs)
        self.batches+=1
        self.peak_gpu_mib=max(self.peak_gpu_mib,torch.cuda.max_memory_allocated()/1024**2)
        if self.batches==self.pilot_iter:
            steps=[int(v['step']) for v in runner.optim_wrapper.optimizer.state.values() if 'step' in v]
            require(steps and min(steps)==max(steps)==self.pilot_iter//self.accumulation,'Pilot optimizer update count differs')
            require(state_digest(runner.model.backbone)==runner.model._backbone_digest,'Pilot changed frozen backbone')
            changes=self.deltas(runner)
            require(changes['neck']>0 and changes['det_head']>0,'Pilot neck/head did not learn')
            if runner.model.neck.mode in ('asf','dysample'):require(changes['new_modules']>0,'New module did not learn')
            path=Path(runner.work_dir)/f'pilot_iter_{self.pilot_iter}.pth'
            runner.save_checkpoint(str(path.parent),filename=path.name,save_optimizer=True,save_param_scheduler=True,by_epoch=False)
            require(path.stat().st_size>1_000_000,'Empty pilot checkpoint')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            require(checkpoint['meta']['iter']==self.pilot_iter,'Pilot checkpoint iteration differs')
            save_json(Path(runner.work_dir).parent/'pilot_gate.json',dict(status='PASS',optimizer_updates=min(steps),
                      micro_iterations=self.batches,module_updates=changes,backbone_unchanged=True,
                      continuation='same Runner, optimizer, dataloader and 10000-update scheduler'))
            print('PILOT PASS; continuing same run to 10000 updates:',runner.model.neck.mode,flush=True)

    def after_train(self,runner):
        require(state_digest(runner.model.backbone)==runner.model._backbone_digest,'Training changed backbone')
        self.stream=dict(micro_iterations=self.batches,source_sha256=self.source.hexdigest(),
                         augmented_inputs_sha256=self.augmentation.hexdigest())
        self.module_updates=self.deltas(runner)
