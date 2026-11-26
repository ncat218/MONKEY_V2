"""Research bundle v1: query-free zoom, seed replicates and PRC2 in ONE allocation.

Development data only: 2,262 remaining-validation sources (Clean + native LR2) and the hashed 500-source
stress cohort of the fusion study. No TEST and no new threshold search: every variant is scored at the
fixed operating points already used by the main flow. The whole-image EDSR / bicubic re-run on fine-tuned
B1 is job 163012 (submit_edsr_recheck.py), so it is not repeated here; only EDSR on zoom crops is.
"""

OUTPUT_NAME = 'research_bundle/v1'
TIME_LIMIT = '1-16:00:00'

SEED_REPLICATE = 43   # second training seed for B1 and B-Clean (sampler order + augmentation RNG)
PRC_SEED = 42         # PRC2 keeps the B1 seed so M-vs-B1 stays paired on one TRAIN stream
TRAIN_ARMS = ('B1_s43', 'BClean_s43', 'PRC2')
ZOOM_MODELS = ('B1', 'B-Clean')
STRESS_ZOOM_MODELS = ('B1',)
MAIN_CONDITIONS = ('clean', 'lr2')
STRESS_CONDITIONS = ('lr4', 'lr8', 'blur1', 'blur2', 'jpeg30', 'jpeg10')

# Fixed operating points: (mask threshold, min text score)
SETTINGS = {'primary': (0.40, 0.60), 'common': (0.30, 0.40)}
NATIVE = (0.30, 0.30)
GRID = tuple((mask, score) for mask in (.30, .20, .15, .10, .05) for score in (.30, .20, .40))
EVAL_MASKS = (.30, .20, .15, .10, .05, .40)
STRESS_MASKS = (.40, .30)

ZOOM = dict(
    tau_ref=8.0,            # LR2-referenced word height (px): zoom handles words below, global above
    k_max=4,                # at most four half-size windows per image (same cap as 2x2 tiling)
    positions=5,            # window origins per axis on a regular grid
    evidence_low=0.20,      # weak-response component threshold on the global P-map
    evidence_strong=0.60,   # components peaking below the primary score cutoff count as weak
    margin_task_px=1.0,     # predictions touching an inner crop border are partial words
    nms_iou=0.5,
    verify_iou=0.3,
    min_window_px=16,
)
ZOOM_VARIANTS = {
    'B1': ('global', 'poly', 'tile2x2', 'tile2x2_union', 'guided', 'guided_union', 'guided_verify', 'guided_edsr',
           'oracle'),
    'B-Clean': ('global', 'tile2x2', 'tile2x2_union', 'guided', 'guided_union', 'guided_verify'),
}
STRESS_VARIANTS = ('global', 'tile2x2', 'tile2x2_union', 'guided', 'guided_verify')
# variant -> (window source, fusion mode, crop restoration)
ZOOM_RULES = {
    'tile2x2': ('tile', 'replace', 'none'),
    'tile2x2_union': ('tile', 'union', 'none'),
    'guided': ('guided', 'replace', 'none'),
    'guided_union': ('guided', 'union', 'none'),
    'guided_verify': ('guided', 'verify', 'none'),
    'guided_edsr': ('guided', 'replace', 'edsr'),
    'oracle': ('oracle', 'replace', 'none'),
}
PLAIN_VARIANTS = ('global', 'poly')

PRC2 = dict(
    weight=1.0, loss='masked probability MSE', teacher='frozen B0', teacher_mask_thr=0.30,
    coverage=0.30, min_kernel_pixels=4,
    selection='GT kernels whose aligned Clean teacher response covers >= c0; no detection matching',
    probe_samples=96, min_mean_selected=2.0, min_active_fraction=0.6, min_alignment_corr=0.6,
    max_teacher_bg_mean=0.25, train_check_microbatches=200, train_min_active_fraction=0.5,
)

# Frozen .40/.60 totals (tp, gt, pred) from fixed diagnostics job 73827; the zoom global pass must match.
REFERENCE_TOTALS = {
    'B1': {'clean': [61252, 78221, 76005], 'lr2': [56360, 78221, 68357]},
    'B-Clean': {'clean': [61650, 78221, 74270], 'lr2': [51432, 78221, 58010]},
}
BCLEAN_CHECKPOINT_SHA = '861bbde2aaf0b83cc98fab148d374276aa84012a7599a43dc1e7a227628aacfc'
FUSION_RUN = 'fusion_study/recovery_v2'
RESTORATION_RECORDS = 'lr2_restoration/run/paired_records.json'
EDSR_RECHECK_JOB = '163012'
# Source flagged by Trí as a newspaper page whose body text is unlabeled (26.8% of B1 FP on his split A).
UNLABELED_SOURCES = ('6e340fcbb1e0a877',)


def setting_key(mask, score):
    return f'{mask:.2f}/{score:.2f}'


def eval_settings():
    keys = [setting_key(m, s) for m, s in GRID]
    keys.append(setting_key(*SETTINGS['primary']))
    return keys


def variants_for(model, condition):
    if condition in MAIN_CONDITIONS:
        return ZOOM_VARIANTS[model]
    if model in STRESS_ZOOM_MODELS:
        return STRESS_VARIANTS
    raise RuntimeError(f'{model} has no zoom variants for {condition}')
