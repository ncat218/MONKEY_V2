"""Geometry and loss for the exploratory, paired Clean/LR2 feature pilot."""
import numpy as np

from rev4 import require
from rev4.prc_geometry import clean_canvas


def teacher_canvas(clean, lr_shape, affine, crop_origin, resize_factors, shape):
    """Warp Clean directly to a 2x canvas; retain the exact student crop/FOV."""
    height, width = shape
    image, valid = clean_canvas(clean, lr_shape, affine, crop_origin,
                                np.asarray(resize_factors)*2, (2*height, 2*width))
    # A student pixel is valid only when all four teacher subpixels are valid.
    valid = valid.reshape(height, 2, width, 2).all(axis=(1, 3))
    return image, valid


def text_mask(shape, polygons, ignored, valid):
    """Rasterize text with subpixel coordinates; exclude ignored/padded pixels."""
    import cv2

    require(tuple(valid.shape) == tuple(shape), 'Distillation valid-mask shape mismatch')
    require(len(polygons) == len(ignored), 'Polygon/ignore lengths differ')
    foreground = np.zeros(shape, dtype=np.uint8)
    excluded = np.zeros(shape, dtype=np.uint8)
    for polygon, ignore in zip(polygons, ignored):
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        require(len(points) >= 3 and np.isfinite(points).all(), 'Invalid distillation polygon')
        points = np.rint(points*256).astype(np.int32)
        cv2.fillPoly(excluded if ignore else foreground, [points], 1, shift=8)
    return (foreground > 0) & (excluded == 0) & valid, (excluded == 0) & valid


def feature_mask(foreground, valid, size):
    import torch.nn.functional as F

    weights = F.interpolate(foreground[None, None].float(), size=size, mode='area')
    allowed = F.interpolate(valid[None, None].float(), size=size, mode='area') >= 1-1e-6
    return weights*allowed


def masked_feature_loss(student, teacher, weights):
    """Channel-RMS-normalized MSE, averaged over channels and valid text area."""
    import torch
    import torch.nn.functional as F

    require(student.ndim == teacher.ndim == weights.ndim == 4 and student.shape[:2] == teacher.shape[:2],
            'Expected paired NCHW features with equal channels')
    require(weights.shape == (student.shape[0], 1, *student.shape[-2:]), 'Invalid feature weights')
    target = F.interpolate(teacher.detach(), size=student.shape[-2:], mode='area')
    normalize = lambda x: x*torch.rsqrt(x.square().mean(dim=1, keepdim=True)+1e-6)
    error = (normalize(student)-normalize(target)).square().mean(dim=1, keepdim=True)
    # Preserve gradients with an empty mask, without diluting tiny text weights.
    return (error*weights).sum()/weights.sum().clamp_min(1e-6)
