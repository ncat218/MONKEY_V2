"""Track B1 geometry and add one masked PRC term to the DBNet head."""
import cv2
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import numpy as np
import torch
from shapely.geometry import Polygon, box, mapping
from mmocr.datasets.transforms import ImgAugWrapper, PackTextDetInputs, RandomCrop, Resize
from mmocr.models.textdet.heads import DBHead
from mmocr.registry import HOOKS, MODELS, TRANSFORMS

from lowres.evaluate_shared import evaluate_image
from lowres.inference import unpack
from rev4 import require
from rev4.b1_plugins import Rev4B1AuditHook
from rev4.prc_geometry import clean_canvas, masked_response_loss


@TRANSFORMS.register_module()
class Rev4PRCImgAug(ImgAugWrapper):
    def _augment_annotations(self, aug, ori_shape, results):
        ok = super()._augment_annotations(aug, ori_shape, results)
        if ok and results['condition'] == 'lr2':
            h, w = ori_shape[:2]
            landmarks = KeypointsOnImage([Keypoint(x=x, y=y) for x, y in ((0, 0), (w, 0), (0, h), (w, h))],
                                         shape=ori_shape)
            origin, x, y, diagonal = aug.augment_keypoints([landmarks])[0].to_xy_array().astype(np.float64)
            require(np.allclose(x+y-origin, diagonal, rtol=0, atol=.01), 'PRC requires affine ImgAug geometry')
            results['prc_affine'] = np.column_stack(((x-origin)/w, (y-origin)/h, origin))
            results['prc_crop_origin'] = (0, 0)
        return ok


@TRANSFORMS.register_module()
class Rev4PRCRandomCrop(RandomCrop):
    def _sample_crop_box(self, img_size, results):
        crop = super()._sample_crop_box(img_size, results)
        if results['condition'] == 'lr2':
            results['prc_crop_origin'] = crop[:2]
        return crop


@TRANSFORMS.register_module()
class Rev4PRCResize(Resize):
    def transform(self, results):
        results = super().transform(results)
        if results['condition'] == 'lr2':
            results['prc_content_shape'] = tuple(results['img'].shape[:2])
        return results


@TRANSFORMS.register_module()
class Rev4PRCPack(PackTextDetInputs):
    def transform(self, results):
        packed = super().transform(results)
        if results['condition'] == 'lr2':
            clean = cv2.imread(results['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(clean is not None, 'Unable to read paired TRAIN clean image')
            lr_shape = tuple(results['ori_shape'])
            require(lr_shape == tuple(max(1, side//2) for side in clean.shape[:2]), 'Unexpected paired LR2 shape')
            image, valid = clean_canvas(clean, lr_shape, results['prc_affine'], results['prc_crop_origin'],
                                        results['scale_factor'], results['prc_content_shape'])
            h, w = valid.shape
            _, pad_h, pad_w = packed['inputs'].shape
            require(h <= pad_h and w <= pad_w, 'PRC content exceeds padded grid')
            teacher_input = torch.zeros_like(packed['inputs'])
            teacher_input[:, :h, :w] = torch.from_numpy(image).permute(2, 0, 1)
            valid_grid = torch.zeros((pad_h, pad_w), dtype=torch.bool)
            valid_grid[:h, :w] = torch.from_numpy(valid)
            sample = packed['data_samples']
            sample.prc_clean = teacher_input
            sample.prc_valid = valid_grid
            sample.set_metainfo({'prc_content_shape': (h, w)})
        return packed


@MODELS.register_module()
class Rev4PRCDBHead(DBHead):
    def bind_teacher(self, teacher, protocol, weight):
        # A frozen auxiliary detector must stay out of state_dict and the optimizer.
        object.__setattr__(self, '_prc_teacher', teacher.eval().requires_grad_(False))
        self.prc_protocol = protocol
        self.prc_weight = weight

    @torch.no_grad()
    def teacher_target(self, sample):
        teacher = self._prc_teacher
        inputs = teacher.data_preprocessor(dict(inputs=[sample.prc_clean], data_samples=None), training=False)
        probability = teacher(**inputs, mode='tensor')[0]
        require(probability.dtype == torch.float32 and probability.shape == sample.prc_valid.shape,
                'Teacher and student must share the padded FP32 grid')
        polygons = sample.gt_instances.polygons
        ignored = sample.gt_instances.ignored.cpu().tolist()
        gt = dict(valid=[], ignored=[])
        for index, (poly, ignore) in enumerate(zip(polygons, ignored)):
            polygon = Polygon(np.asarray(poly).reshape(-1, 2))
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty or polygon.area <= 0:
                continue
            gt['ignored' if ignore else 'valid'].append(dict(ann_id=index, geometry=mapping(polygon),
                                                            original_height=polygon.bounds[3]-polygon.bounds[1]))
        h, w = sample.prc_content_shape
        masked = probability.masked_fill(~sample.prc_valid, 0.)
        prediction = unpack(teacher.det_head.postprocessor.get_text_instances(masked, sample.new()), (1., 1.))
        matched = evaluate_image(gt, mapping(box(0, 0, w, h)), prediction, self.prc_protocol)['matched_ann_ids']
        size = tuple(probability.shape)
        kernels, _ = self.module_loss._generate_kernels(size, [polygons[i] for i in matched],
                                                        self.module_loss.shrink_ratio)
        keep = self.module_loss._generate_effective_mask(size, [p for p, flag in zip(polygons, ignored) if flag])
        labels = torch.as_tensor(kernels, dtype=torch.long, device=probability.device)
        valid = torch.as_tensor(keep > 0, device=probability.device) & sample.prc_valid
        labels = labels.masked_fill(~valid, 0)
        sizes = torch.bincount(labels.reshape(-1), minlength=len(matched)+1)
        mask = (labels > 0) & (sizes[labels] >= 4)
        return probability, mask, len(matched)

    def loss(self, x, batch_data_samples):
        outputs = self(x, batch_data_samples, mode='both')
        losses = self.module_loss(outputs[:3], batch_data_samples)
        require([s.get('condition') for s in batch_data_samples] == ['clean', 'lr2'], 'PRC requires the B1 microbatch')
        student = outputs[3][1]
        target, mask, matches = self.teacher_target(batch_data_samples[1])
        require(student.shape == target.shape, 'PRC probability grids differ')
        losses['loss_prc'] = self.prc_weight*masked_response_loss(student, target, mask)
        losses['prc_pixels'] = mask.sum().float()
        losses['prc_teacher_matches'] = student.new_tensor(float(matches))
        return losses


@HOOKS.register_module()
class Rev4PRCAuditHook(Rev4B1AuditHook):
    # Run before MMEngine's logger resets the CUDA peak counter.
    priority = 'ABOVE_NORMAL'

    def __init__(self):
        self.peak_mib = 0.
        self.active_microbatches = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        self.active_microbatches += int(outputs['prc_pixels'].item() > 0)
        self.peak_mib = max(self.peak_mib, torch.cuda.max_memory_allocated()/1024**2)

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint['prc_audit'] = self.active_microbatches, self.peak_mib

    def after_load_checkpoint(self, runner, checkpoint):
        self.active_microbatches, self.peak_mib = checkpoint.get('prc_audit', (0, 0.))
