# Rev.4 B0/B1 calibration

2,262 remaining validation sources paired across Clean and LR2. Threshold selection uses mean micro-Hmean over the two conditions.

| Model | Mode | Mask | Score | Condition | Precision | Recall | Hmean | TP | FP | FN |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| B0 | native | 0.30 | 0.30 | clean | 76.46% | 66.31% | 71.03% | 51,871 | 15,971 | 26,350 |
| B0 | native | 0.30 | 0.30 | lr2 | 79.50% | 47.90% | 59.78% | 37,466 | 9,661 | 40,755 |
| B0 | calibrated | 0.30 | 0.40 | clean | 82.03% | 66.25% | 73.30% | 51,820 | 11,352 | 26,401 |
| B0 | calibrated | 0.30 | 0.40 | lr2 | 87.94% | 47.81% | 61.95% | 37,399 | 5,129 | 40,822 |
| B1 | native | 0.30 | 0.30 | clean | 66.02% | 78.92% | 71.89% | 61,734 | 31,781 | 16,487 |
| B1 | native | 0.30 | 0.30 | lr2 | 66.78% | 73.28% | 69.88% | 57,323 | 28,512 | 20,898 |
| B1 | calibrated | 0.30 | 0.40 | clean | 76.32% | 78.82% | 77.55% | 61,655 | 19,132 | 16,566 |
| B1 | calibrated | 0.30 | 0.40 | lr2 | 77.62% | 73.10% | 75.29% | 57,178 | 16,487 | 21,043 |

| B1 − B0 Hmean | Clean (pp) | LR2 (pp) | Mean (pp) |
|---|---:|---:|---:|
| native | +0.87 | +10.10 | +5.49 |
| calibrated | +4.25 | +13.35 | +8.80 |

B1 native counts reproduce job 12187. These validation scores also support threshold selection.
