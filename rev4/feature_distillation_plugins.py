"""Training-only teacher and mask, with an unchanged DBNet inference graph."""
import cv2
import numpy as np
import torch

from mmocr.datasets.transforms import PackTextDetInputs
from mmocr.models.textdet.heads import DBHead
from mmocr.registry import HOOKS, MODELS, TRANSFORMS

from rev4 import require
from rev4.b1_plugins import Rev4B1AuditHook
from rev4.feature_distillation import feature_mask, masked_feature_loss, teacher_canvas, text_mask


@TRANSFORMS.register_module()
class FeaturePilotPack(PackTextDetInputs):
    def transform(self, results):
        packed = super().transform(results)
        if results['condition'] != 'lr2':
            return packed
        clean = cv2.imread(results['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        require(clean is not None, 'Unable to decode paired TRAIN Clean')
        lr_shape = tuple(results['ori_shape'])
        require(lr_shape == tuple(max(1, side//2) for side in clean.shape[:2]), 'Unexpected native LR2 dimensions')
        image, valid = teacher_canvas(clean, lr_shape, results['prc_affine'], results['prc_crop_origin'],
                                      results['scale_factor'], results['prc_content_shape'])
        sample = packed['data_samples']
        _, pad_h, pad_w = packed['inputs'].shape
        h, w = valid.shape
        require(h <= pad_h and w <= pad_w, 'Distillation content exceeds padding')
        teacher = packed['inputs'].new_zeros((3, 2*pad_h, 2*pad_w))
        teacher[:, :2*h, :2*w] = torch.from_numpy(image).permute(2, 0, 1)
        valid_grid = np.zeros((pad_h, pad_w), dtype=bool)
        valid_grid[:h, :w] = valid
        foreground, allowed = text_mask((pad_h, pad_w), sample.gt_instances.polygons,
                                        sample.gt_instances.ignored.cpu().tolist(), valid_grid)
        sample.fd_clean = teacher
        sample.fd_foreground = torch.from_numpy(foreground)
        sample.fd_valid = torch.from_numpy(allowed)
        return packed


@MODELS.register_module()
class FeaturePilotHead(DBHead):
    def bind_teacher(self, teacher, weight):
        # Do not register the teacher as a child module or serialize its weights.
        object.__setattr__(self, '_fd_teacher', teacher.eval().requires_grad_(False))
        self.fd_weight = float(weight)

    def loss(self, x, batch_data_samples):
        outputs = self(x, batch_data_samples, mode='both')
        losses = self.module_loss(outputs[:3], batch_data_samples)
        require([s.get('condition') for s in batch_data_samples] == ['clean', 'lr2'], 'Unbalanced pilot batch')
        sample = batch_data_samples[1]
        teacher = self._fd_teacher
        with torch.no_grad():
            processed = teacher.data_preprocessor(dict(inputs=[sample.fd_clean], data_samples=None), training=False)
            target = teacher.extract_feat(processed['inputs'])
        require(isinstance(x, torch.Tensor) and isinstance(target, torch.Tensor), 'Require fused FPNC tensors')
        student = x[1:2]
        require(target.shape[-2:] == tuple(2*v for v in student.shape[-2:]), 'Teacher FPNC must have 2x spatial size')
        weights = feature_mask(sample.fd_foreground, sample.fd_valid, student.shape[-2:])
        kd = masked_feature_loss(student, target, weights)
        losses.update(loss_feature_kd=self.fd_weight*kd, fd_raw_mse=kd.detach(),
                      fd_weight_sum=weights.sum().detach(), fd_active=(weights.sum() > 0).float().detach())
        # Used once in the GPU probe to verify the KD path independently of detection loss.
        if getattr(self, 'fd_probe', False):
            object.__setattr__(self, '_fd_probe_feature', student)
        return losses


@HOOKS.register_module()
class FeaturePilotAuditHook(Rev4B1AuditHook):
    priority = 'ABOVE_NORMAL'

    def __init__(self):
        self.peak_mib = 0.
        self.active = 0
        self.batches = 0
        self.kd_sum = 0.

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        self.batches += 1
        self.active += int(outputs.get('fd_active', 0) > 0)
        self.kd_sum += float(outputs.get('fd_raw_mse', 0))
        self.peak_mib = max(self.peak_mib, torch.cuda.max_memory_allocated()/1024**2)
        if hasattr(runner.model.det_head, '_fd_teacher') and self.batches == 100:
            require(self.active > 0, 'No valid distillation mask in the first 100 microbatches; stopping')
