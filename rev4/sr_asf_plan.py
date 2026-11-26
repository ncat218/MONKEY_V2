"""Factorial pretrained SR x completed ASF inference; no additional training."""
import copy
from rev4.sr_plan import RECIPE as SR_RECIPE

ASF_SHA = '9cae7ad418cc906edb56667c03c0d5f9e25f02b7ad929e4024ff03ebaec38059'
BASELINE_JOB = '163012'
CELLS = tuple(f'{detector}/{branch}' for detector in ('B1','ASF')
              for branch in ('native','bicubic_x2','edsr_x2'))
COMPARISONS = (
    ('ASF/native','B1/native'), ('B1/edsr_x2','B1/native'),
    ('ASF/edsr_x2','B1/native'), ('ASF/edsr_x2','ASF/native'),
    ('ASF/edsr_x2','B1/edsr_x2'), ('ASF/bicubic_x2','ASF/native'),
    ('ASF/edsr_x2','ASF/bicubic_x2'), ('ASF/bicubic_x2','B1/bicubic_x2'))
RECIPE = copy.deepcopy(SR_RECIPE)
RECIPE.update(detector='completed ASF 10000-extra-update checkpoint from B1; all weights frozen',
    detector_sha256=ASF_SHA, baseline_job=BASELINE_JOB, cells=list(CELLS),
    execution='after successful baseline163012; reuse exact completed B1 records, infer only three ASF inputs',
    interaction='(EDSR+ASF - ASF) - (EDSR+B1 - B1), paired-source intervals',
    timing='within-run SR/detector timing; B1 reference comes from another allocation, cross-run timings descriptive')
