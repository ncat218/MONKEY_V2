# Paired B0 / B-Clean / B1 adaptation EDA

Development data only. All per-image counts must reproduce the saved calibration counts.
Native = 0.30/0.30; common = 0.30/0.40; calibrated = each model's selected original 15-pair grid.

| Model | Mode | Clean Hmean | LR2 Hmean | Clean FP | LR2 FP |
|---|---|---:|---:|---:|---:|
| B0 | native | 71.03% | 59.78% | 15971 | 9661 |
| B0 | common | 73.30% | 61.95% | 11352 | 5129 |
| B0 | calibrated | 73.30% | 61.95% | 11352 | 5129 |
| B-Clean | native | 73.79% | 69.46% | 27537 | 20662 |
| B-Clean | common | 78.79% | 74.36% | 16814 | 10457 |
| B-Clean | calibrated | 78.79% | 74.36% | 16814 | 10457 |
| B1 | native | 71.89% | 69.88% | 31781 | 28512 |
| B1 | common | 77.55% | 75.29% | 19132 | 16487 |
| B1 | calibrated | 77.55% | 75.29% | 19132 | 16487 |

| Comparison (new minus old) | Condition | Hmean difference pp [95% CI] | Rescued | Regressed | Persistent FN |
|---|---|---:|---:|---:|---:|
| B0 → B1 | clean | +4.250 [+3.079, +5.384] | 11672 | 1837 | 14729 |
| B0 → B1 | lr2 | +13.346 [+10.254, +16.329] | 20891 | 1112 | 19931 |
| B0 → B-Clean | clean | +5.487 [+4.489, +6.458] | 11722 | 1772 | 14679 |
| B0 → B-Clean | lr2 | +12.412 [+10.608, +14.398] | 16480 | 1398 | 24342 |
| B-Clean → B1 | clean | -1.236 [-2.056, -0.641] | 1536 | 1651 | 14915 |
| B-Clean → B1 | lr2 | +0.933 [-1.433, +3.142] | 6150 | 1453 | 19590 |

CI: 2,000 paired source bootstrap replicates, seed 20261107. Post-selection descriptive intervals, not training-seed uncertainty.
FP correspondence matches FP polygons across models at IoU ≥ 0.5 for description; it does not change evaluation.
P-map statistics and geometric categories are diagnostic proxies. Inspect selected image/P-map pairs before inferring causes.
Two examples per height bin and transition are chosen deterministically, including regressions; this is not a random prevalence estimate.
Raw prediction dimensions and actual backbone/FPNC shapes are in input_shapes.csv. EDA timing is not a latency benchmark.

Artifacts: paired_transitions.csv, fp_analysis.csv, model_metrics.csv, stratified_metrics.csv,
failure_distributions.json, selected_examples.json and examples/ (paired image/P-map crops).
