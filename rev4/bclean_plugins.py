"""100% Clean at the same source draw slots and optimization budget as B1."""
import torch
from torch.utils.data import Sampler

from mmocr.registry import DATA_SAMPLERS, HOOKS
from rev4 import require
from rev4.b1_plugins import Rev4B1AuditHook
from rev4.training_data import RECIPE, TRAIN_SOURCES, balanced_indices


@DATA_SAMPLERS.register_module()
class CleanControlSampler(Sampler):
    def __init__(self, dataset, seed=None):
        require(len(dataset) == 2*TRAIN_SOURCES, 'Expected two source-identical Clean draw groups')
        for index, slot in ((0, 0), (TRAIN_SOURCES-1, 0), (TRAIN_SOURCES, 1), (2*TRAIN_SOURCES-1, 1)):
            item = dataset.get_data_info(index)
            require(item['condition'] == 'clean' and item['clean_draw_slot'] == slot, 'Invalid Clean draw group')
        self.count = TRAIN_SOURCES
        self.seed = RECIPE['seed'] if seed is None else seed

    def __iter__(self):
        return balanced_indices(self.count, self.seed)

    def __len__(self):
        return 2*self.count


@HOOKS.register_module()
class CleanControlAuditHook(Rev4B1AuditHook):
    priority = 'ABOVE_NORMAL'

    def __init__(self):
        self.batches = 0
        self.images = 0
        self.peak_mib = 0.

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        samples = data_batch['data_samples']
        require([s.get('condition') for s in samples] == ['clean', 'clean'], 'B-Clean saw a non-Clean image')
        require([s.get('clean_draw_slot') for s in samples] == [0, 1], 'Clean draw sequence differs from B1 slots')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        self.batches += 1
        self.images += len(data_batch['data_samples'])
        self.peak_mib = max(self.peak_mib, torch.cuda.max_memory_allocated()/1024**2)

