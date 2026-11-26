"""Frozen transfer of Tri's box-quality v1, after the EDSR x ASF reference."""
import copy
from rev4.sr_asf_plan import RECIPE as PARENT_RECIPE

PARENT_JOB = '163064'
VENDOR_COMMIT = '7517ce53f86962e07fd440397f0c525faea45c00'
ASSETS = 'docs/handoff/artifacts/tri_quality_frozen'
ARMS = ('F', 'RAW', 'Q')
DETECTORS = ('B1', 'ASF')
BRANCHES = ('native', 'bicubic_x2', 'edsr_x2')
CELLS = tuple(f'{d}/{b}/{s}' for d in DETECTORS for b in BRANCHES for s in ARMS)
CONTRASTS = tuple((f'{d}/{b}/Q', f'{d}/{b}/{control}')
    for d in DETECTORS for b in BRANCHES for control in ('F', 'RAW')) + (
    ('ASF/native/Q', 'B1/native/F'), ('ASF/native/Q', 'B1/native/Q'),
    ('ASF/edsr_x2/Q', 'ASF/native/Q'), ('ASF/edsr_x2/Q', 'B1/edsr_x2/Q'),
    ('ASF/edsr_x2/Q', 'ASF/edsr_x2/F'), ('ASF/edsr_x2/Q', 'B1/native/F'))
# Deduplicate a contrast that also appears in the per-cell quality effects.
CONTRASTS = tuple(dict.fromkeys(CONTRASTS))
RECIPE = copy.deepcopy(PARENT_RECIPE)
RECIPE.update(stage='frozen box-quality transfer v1', parent_job=PARENT_JOB,
    scorer_repository='https://github.com/minhtridaoo/monkeyocrv2-mix', scorer_commit=VENDOR_COMMIT,
    scorer='exact pretrained 13-feature logistic v1; no fit, no cutoff selection',
    candidate_geometry=dict(mask_thr=.4, unclip_ratio=1.5, min_text_width=5, epsilon_ratio=.01,
                            text_repr_type='quad', max_candidates=3000, raw_floor=.20),
    thresholds=dict(F=.60, RAW=.5976406401908236, Q=.3315908613227198),
    operating_points='only .40 geometry; historical .30/.40 is not a scorer configuration',
    cells=list(CELLS), contrasts=[list(x) for x in CONTRASTS],
    execution='independent successor after163064; rerun P-maps for frozen features; exact F parity for every cell',
    cohort='same2262 paired sources and fixed500 stress sources; A/B descriptive subgroups, no selection',
    latency_sources=20, latency_passes=2,
    timing='dedicated interleaved F/Q passes; restoration measured once per image and shared; no production claim',
    no_training=True, no_test=True)
