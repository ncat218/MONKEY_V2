# PR, quad postprocessing and paired error diagnosis

Exploratory analysis on the same 2,262 remaining-validation sources previously used for model selection, calibration and EDA. This is not a fresh holdout. S1/S2 were already viewed in pilot work. TEST was not read. No training was run.

Installed MMOCR DBPostprocessor: `/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/environment_and_models__phase1_verified_stack/upstream_MonkeyOCRv2/detection/mmocr/mmocr/mmocr/models/textdet/postprocessors/db_postprocessor.py` (SHA-256 `9bfff8aaaca103dd08d7565f2b7a5410ab20a6427fe1f0baad5a5d04d207c8b8`). The audited inference path is `quad`; the original protocol/evaluator were not changed.

## Existing calibrated baseline and exploratory selection

| Model | Setting selected within this exploratory grid | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP | LR2 recall <4 px |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | epsilon_005/score_0.45; mask 0.30, unclip 1.50, min width 5, epsilon 0.005, score 0.45 | 74.09% | 62.38% | 90.16% | 47.68% | 4,069 | 1.01% |
| B-Clean | mask_040/score_0.60; mask 0.40, unclip 1.50, min width 5, epsilon 0.010, score 0.60 | 80.86% | 75.51% | 88.66% | 65.75% | 6,578 | 15.96% |
| B1 | mask_040/score_0.60; mask 0.40, unclip 1.50, min width 5, epsilon 0.010, score 0.60 | 79.43% | 76.90% | 82.45% | 72.05% | 11,997 | 45.79% |

Selection rule: maximize the unweighted mean of Clean and LR2 micro-Hmean over the same grid for each model; the original base `.30/.40` setting wins exact ties. This is exploratory selection on already-used validation, not an independent performance claim.

## Change from original calibrated `.30/.40`

| Model | Condition | Δ Hmean (pp) | Paired source-bootstrap 95% CI |
|---|---|---:|---|
| B0 | clean | +0.796 | [+0.691, +0.907] |
| B0 | lr2 | +0.430 | [+0.334, +0.524] |
| B-Clean | clean | +2.071 | [+1.474, +2.626] |
| B-Clean | lr2 | +1.150 | [+0.669, +1.619] |
| B1 | clean | +1.882 | [+1.242, +2.465] |
| B1 | lr2 | +1.610 | [+1.006, +2.301] |

## Candidate and GT diagnosis

The taxonomy CSV contains automatic geometric proxies, not verified causes. The human review sheet is pending and must be reviewed before treating categories as visual findings.

- Sampled review cards: 160; status: PENDING.
- FP proxy categories and GT transition counts are in `diagnosis_summary.json`.
- Per-prediction/GT rows are in `fp_taxonomy_proxies.csv` and `gt_diagnosis.csv`.
- Human labels and reviewer notes belong in `human_review.csv`; preserve unreviewed labels as blank.

## Artifacts

- `pr_curves.csv`: base quad PR/Hmean and FP versus <4 px recall over score thresholds.
- `threshold_frontier.csv`: all exploratory geometry and score combinations.
- `quad_filter_audit.csv`: contour cap, approximation, score, unclip and min-width flow counters.
- `review/`: stratified image/P-map pairs for human inspection.
- `run_identity.json`: source, checkpoint, data and installed postprocessor hashes.

No training or TEST evaluation was performed. Do not replace the frozen `.30/.40` reports with these exploratory results.

Recovery of job 54308: original source-count matrices and sweep CSVs were reused after hash and per-image baseline checks. Only sampled review images required new forward passes. Source-paired CIs are descriptive after selection and do not measure training-seed variation.
