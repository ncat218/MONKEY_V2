"""MMOCR transforms and sampling for the locked Clean/LR2 training recipe."""
import math

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Sampler
from mmengine.hooks import Hook
from mmocr.datasets.transforms.loading import LoadImageFromFile
from mmocr.registry import DATA_SAMPLERS, HOOKS, TRANSFORMS

from rev4 import require
from rev4.training_data import RECIPE, TRAIN_SOURCES, balanced_indices, native_lr2


@TRANSFORMS.register_module()
class Rev4LoadCleanOrLR2(LoadImageFromFile):
    def transform(self, results):
        condition = results['condition']
        if condition == 'clean':
            results = super().transform(results)
            require(results is not None, 'Clean decode failed')
        elif condition == 'lr2':
            with Image.open(results['img_path']) as image:
                bgr = np.asarray(native_lr2(image))[:, :, ::-1].copy()
            results.update(img=bgr, img_shape=bgr.shape[:2], ori_shape=bgr.shape[:2])
        else:
            raise ValueError(f'Unexpected B1 training condition {condition}')
        require(tuple(results['img'].shape[:2]) == (results['height'], results['width']),
                f'Training image and annotation dimensions differ: {results["sample_id"]}')
        return results


@DATA_SAMPLERS.register_module()
class Rev4BalancedLR2Sampler(Sampler):
    """Independently shuffle each equal group and alternate one draw from each."""
    def __init__(self, dataset, seed=None):
        require(len(dataset) == 2*TRAIN_SOURCES, 'TRAIN dataset must have two complete groups')
        require(dataset.get_data_info(0)['condition'] == 'clean' and
                dataset.get_data_info(TRAIN_SOURCES-1)['condition'] == 'clean' and
                dataset.get_data_info(TRAIN_SOURCES)['condition'] == 'lr2' and
                dataset.get_data_info(len(dataset)-1)['condition'] == 'lr2',
                'TRAIN dataset group order differs from the sampler layout')
        self.seed = RECIPE['seed'] if seed is None else seed
        self.count = len(dataset)//2

    def __iter__(self):
        return balanced_indices(self.count, self.seed)

    def __len__(self):
        return 2*self.count


@HOOKS.register_module()
class Rev4B1AuditHook(Hook):
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        conditions = [sample.get('condition') for sample in data_batch['data_samples']]
        require(conditions == ['clean', 'lr2'], f'Invalid B1 microbatch conditions {conditions}')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for name, value in (outputs or {}).items():
            if isinstance(value, (int, float)):
                require(math.isfinite(value), f'Nonfinite training log {name}')
            elif isinstance(value, torch.Tensor):
                require(torch.isfinite(value).all(), f'Nonfinite training tensor {name}')
