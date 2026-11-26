"""PRC2: paired response consistency restricted to GT kernels the aligned Clean teacher covers.

PRC v1 (job 12518) selected kernels through teacher detection + polygon matching and was active in
only 608/20,000 microbatches (logged teacher matches ~0). PRC2 keeps the same Clean->LR2 alignment
transforms and masked MSE, but selects kernels directly from the teacher probability map (kernel
coverage, the protocol's own C_GT measure), so the target no longer depends on the postprocessor
or the evaluator.
"""
import numpy as np
import torch

from mmocr.models.textdet.heads import DBHead
from mmocr.registry import HOOKS, MODELS

import rev4.prc_plugins  # noqa: F401  (registers the aligned Rev4PRC* transforms)
from rev4 import require
from rev4.b1_plugins import Rev4B1AuditHook
from rev4.prc_geometry import masked_response_loss


def kernel_selection(probability, labels, valid, mask_thr, coverage, min_pixels):
    """Select kernel instances whose teacher coverage (P > mask_thr) reaches ``coverage``.

    probability: (H, W) float tensor; labels: (H, W) long tensor of kernel ids (0 = none);
    valid: (H, W) bool tensor. Returns (mask, selected_count, eligible_count).
    """
    labels = labels.masked_fill(~valid, 0)
    count = int(labels.max().item()) if labels.numel() else 0
    if count == 0:
        return torch.zeros_like(valid), 0, 0
    flat = labels.reshape(-1)
    sizes = torch.bincount(flat, minlength=count+1)
    hits = torch.bincount(flat[(probability > mask_thr).reshape(-1)], minlength=count+1)
    eligible = sizes >= min_pixels
    eligible[0] = False
    chosen = eligible & (hits.to(torch.float64) >= coverage*sizes.to(torch.float64))
    mask = chosen[labels] & (labels > 0)
    return mask, int(chosen.sum().item()), int(eligible.sum().item())


def teacher_kernels(module_loss, sample, size):
    polygons = sample.gt_instances.polygons
    ignored = sample.gt_instances.ignored.cpu().tolist()
    valid = [p for p, flag in zip(polygons, ignored) if not flag]
    kernels, _ = module_loss._generate_kernels(size, valid, module_loss.shrink_ratio)
    keep = module_loss._generate_effective_mask(size, [p for p, flag in zip(polygons, ignored) if flag])
    return kernels, keep


@MODELS.register_module()
class Rev4PRC2DBHead(DBHead):
    def bind_teacher(self, teacher, recipe):
        # Frozen auxiliary detector stays out of state_dict and the optimizer.
        object.__setattr__(self, '_prc2_teacher', teacher.eval().requires_grad_(False))
        self.prc2 = dict(recipe)

    @torch.no_grad()
    def teacher_target(self, sample):
        teacher = self._prc2_teacher
        inputs = teacher.data_preprocessor(dict(inputs=[sample.prc_clean], data_samples=None), training=False)
        probability = teacher(**inputs, mode='tensor')[0]
        require(probability.dtype == torch.float32 and probability.shape == sample.prc_valid.shape,
                'Teacher and student must share the padded FP32 grid')
        size = tuple(probability.shape)
        kernels, keep = teacher_kernels(self.module_loss, sample, size)
        labels = torch.as_tensor(np.asarray(kernels), dtype=torch.long, device=probability.device)
        valid = torch.as_tensor(np.asarray(keep) > 0, device=probability.device) & sample.prc_valid
        mask, selected, eligible = kernel_selection(probability, labels, valid, self.prc2['teacher_mask_thr'],
                                                    self.prc2['coverage'], self.prc2['min_kernel_pixels'])
        return probability, mask, selected, eligible

    def loss(self, x, batch_data_samples):
        outputs = self(x, batch_data_samples, mode='both')
        losses = self.module_loss(outputs[:3], batch_data_samples)
        require([s.get('condition') for s in batch_data_samples] == ['clean', 'lr2'], 'PRC2 requires the B1 microbatch')
        student = outputs[3][1]
        target, mask, selected, eligible = self.teacher_target(batch_data_samples[1])
        require(student.shape == target.shape, 'PRC2 probability grids differ')
        losses['loss_prc2'] = self.prc2['weight']*masked_response_loss(student, target, mask)
        losses['prc2_pixels'] = mask.sum().float()
        losses['prc2_selected'] = student.new_tensor(float(selected))
        losses['prc2_eligible'] = student.new_tensor(float(eligible))
        return losses


@HOOKS.register_module()
class Rev4PRC2AuditHook(Rev4B1AuditHook):
    # Run before MMEngine's logger resets the CUDA peak counter.
    priority = 'ABOVE_NORMAL'

    def __init__(self, check_after=200, min_active=0.5):
        self.check_after, self.min_active = int(check_after), float(min_active)
        self.batches = 0
        self.active = 0
        self.selected = 0.
        self.eligible = 0.
        self.peak_mib = 0.

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        self.batches += 1
        self.active += int(float(outputs['prc2_pixels']) > 0)
        self.selected += float(outputs['prc2_selected'])
        self.eligible += float(outputs['prc2_eligible'])
        self.peak_mib = max(self.peak_mib, torch.cuda.max_memory_allocated()/1024**2)
        if self.batches == self.check_after:
            require(self.active >= self.min_active*self.batches,
                    f'PRC2 active in only {self.active}/{self.batches} early microbatches; stopping')

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint['prc2_audit'] = [self.batches, self.active, self.selected, self.eligible, self.peak_mib]

    def after_load_checkpoint(self, runner, checkpoint):
        self.batches, self.active, self.selected, self.eligible, self.peak_mib = checkpoint.get(
            'prc2_audit', [0, 0, 0., 0., 0.])
