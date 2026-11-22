# Exploratory LR2 feature-distillation pilot

Post-hoc extension, not a replacement for frozen Rev.4. Same B1 initialization and 1,000 extra optimizer updates per arm.
2,262 remaining-validation sources. No TEST, no new threshold search. One training seed.

| Model | Mask/score | Clean Hmean | LR2 Hmean | LR2 FP |
|---|---|---:|---:|---:|
| B1 | 0.30/0.30 | 71.89% | 69.88% | 28,512 |
| B1 | 0.30/0.40 | 77.55% | 75.29% | 16,487 |
| continued | 0.30/0.30 | 72.17% | 70.13% | 28,052 |
| continued | 0.30/0.40 | 77.33% | 75.24% | 16,766 |
| feature_kd | 0.30/0.30 | 72.08% | 70.11% | 28,163 |
| feature_kd | 0.30/0.40 | 77.32% | 75.24% | 16,825 |

| Comparison at fixed .30/.40 | Clean delta pp (95% CI) | LR2 delta pp (95% CI) |
|---|---:|---:|
| continued_minus_B1 | -0.218 [-0.403, -0.015] | -0.047 [-0.244, +0.136] |
| feature_kd_minus_B1 | -0.232 [-0.417, -0.034] | -0.048 [-0.256, +0.135] |
| feature_kd_minus_continued | -0.015 [-0.044, +0.014] | -0.000 [-0.035, +0.035] |

KD active in 1903/2000 microbatches. See pilot_summary.json for bins, training cost and native timing.
CIs are descriptive paired-source intervals on development validation and exclude training-seed uncertainty.
Native timing is a rough end-to-end detector measure, not a controlled deployment latency benchmark.
