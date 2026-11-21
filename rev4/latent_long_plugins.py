"""Long-run audits: whole data stream, actual module updates and GPU peak."""
import hashlib
import torch
from mmocr.registry import HOOKS

from rev4 import require
from rev4.latent_plugins import LatentAuditHook


@HOOKS.register_module()
class LatentLongAuditHook(LatentAuditHook):
    def before_train(self,runner):
        super().before_train(runner)
        self.source_digest=hashlib.sha256()
        self.augmentation_digest=hashlib.sha256()
        self.peak_gpu_mib=0.
        self.initial={name:p.detach().cpu().clone() for name,p in runner.model.named_parameters()
                      if name.startswith(('feature_adapter.','det_head.'))}

    def before_train_iter(self,runner,batch_idx,data_batch=None):
        super().before_train_iter(runner,batch_idx,data_batch)
        for sample in data_batch['data_samples']:
            self.source_digest.update((sample.get('sample_id')+'\n').encode('utf-8'))
        for tensor in data_batch['inputs']:
            self.augmentation_digest.update(str(tuple(tensor.shape)).encode('ascii'))
            self.augmentation_digest.update(tensor.cpu().contiguous().numpy().tobytes())

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        super().after_train_iter(runner,batch_idx,data_batch,outputs)
        self.peak_gpu_mib=max(self.peak_gpu_mib,torch.cuda.max_memory_allocated()/1024**2)

    def after_train(self,runner):
        super().after_train(runner)
        self.stream_receipt=dict(micro_iterations=self.batches,source_sha256=self.source_digest.hexdigest(),
                                 augmented_inputs_sha256=self.augmentation_digest.hexdigest())
        self.module_updates={}
        for prefix in ('feature_adapter.','det_head.'):
            squared=0.
            for name,p in runner.model.named_parameters():
                if name.startswith(prefix):squared+=float((p.detach().cpu()-self.initial[name]).double().square().sum())
            self.module_updates[prefix.rstrip('.')]=squared**.5
        require(self.module_updates['det_head']>0,'DBHead did not update')
        if runner.model.adapter_kind!='head':
            require(self.module_updates['feature_adapter']>0,'Adapter did not update')
