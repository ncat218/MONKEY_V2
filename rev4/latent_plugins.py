"""Frozen backbone/FPNC, trainable DBHead and optional residual refiner."""
import torch
import hashlib
from mmengine.hooks import Hook
from mmocr.models.textdet.detectors import DBNet
from mmocr.registry import HOOKS, MODELS

from rev4 import require
from rev4.b1_plugins import Rev4B1AuditHook
from rev4.latent_modules import build_refiner, frozen_digest


@MODELS.register_module()
class LatentPilotDBNet(DBNet):
    def __init__(self,adapter_kind='head',**kwargs):
        super().__init__(**kwargs)
        self.adapter_kind=adapter_kind
        self.feature_adapter=build_refiner(adapter_kind)
        for component in (self.backbone,self.neck):
            component.requires_grad_(False)
            component.eval()

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        self.neck.eval()
        return self

    def extract_feat(self,inputs):
        with torch.no_grad():
            feature=super().extract_feat(inputs)
        if not isinstance(feature,torch.Tensor) or feature.ndim!=4 or feature.shape[1]!=256:
            raise ValueError('Latent pilot requires the single 256-channel FPNC tensor')
        return self.feature_adapter(feature)


@HOOKS.register_module()
class LatentAuditHook(Rev4B1AuditHook):
    def __init__(self):
        self.batches=0
        self.draws=[]
        self.start_digest=None

    def before_train(self,runner):
        self.start_digest=frozen_digest(runner.model)
        require(self.start_digest==runner.model._initial_frozen_digest,'Runner changed B1 base initialization')
        require(all(torch.equal(v,runner.model.det_head.state_dict()[k].detach().cpu())
                    for k,v in runner.model._initial_head_state.items()),'Runner changed initial B1 DBHead')

    def before_train_iter(self,runner,batch_idx,data_batch=None):
        super().before_train_iter(runner,batch_idx,data_batch)
        model=runner.model
        require(not model.backbone.training and not model.neck.training,'Frozen components entered train mode')
        require(all(not p.requires_grad and p.grad is None for c in (model.backbone,model.neck) for p in c.parameters()),
                'Frozen base received gradients')
        if self.batches<8:
            digest=hashlib.sha256()
            for tensor in data_batch['inputs']:
                digest.update(tensor.cpu().contiguous().numpy().tobytes())
            self.draws.append(dict(sample_ids=[s.get('sample_id') for s in data_batch['data_samples']],
                                   augmented_input_sha256=digest.hexdigest()))

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        super().after_train_iter(runner,batch_idx,data_batch,outputs)
        self.batches+=1

    def after_train(self,runner):
        require(frozen_digest(runner.model)==self.start_digest,'Frozen base parameters/buffers changed')
