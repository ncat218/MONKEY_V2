"""Alignment and masked response loss for the single PRC intervention."""
import numpy as np


def clean_canvas(clean, lr_shape, lr_to_aug, crop_origin, resize_factors, shape):
    import cv2

    height, width = shape
    lr_height, lr_width = lr_shape
    lr_to_grid = lr_to_aug.copy()
    lr_to_grid[:, 2] -= crop_origin
    lr_to_grid *= np.asarray(resize_factors)[:, None]
    # ImgAug landmarks and GT use image edges. OpenCV warps pixel centers.
    lr_to_grid[:, 2] += lr_to_grid[:, :2] @ np.full(2, .5) - .5
    clean_to_lr = np.asarray([lr_width/clean.shape[1], lr_height/clean.shape[0]])
    clean_to_grid = lr_to_grid.copy()
    clean_to_grid[:, :2] *= clean_to_lr
    clean_to_grid[:, 2] += lr_to_grid[:, :2] @ ((clean_to_lr-1)/2)
    image = cv2.warpAffine(clean, clean_to_grid, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.ones(lr_shape, dtype=np.uint8), lr_to_grid, (width, height),
                           flags=cv2.INTER_NEAREST).astype(bool)
    image[~valid] = 0
    return image, valid


def masked_response_loss(student, teacher, mask):
    weights = mask.to(student.dtype)
    return ((student-teacher.detach()).square()*weights).sum()/weights.sum().clamp_min(1)
