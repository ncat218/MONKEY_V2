# B1-cal LR2 residual diagnostics

2,262 paired remaining-validation sources. Mask 0.30, score 0.40. Every task reproduces job 12396 counts.

Clean TP → LR2 FN **6,454**. Misses shared by Clean and LR2 **14,589**. Clean FN → LR2 TP **1,977**.

| Reason among Clean TP → LR2 FN | c0=0.30 | c0=0.50 |
|---|---:|---:|
| ambiguous_matching | 2 | 2 |
| ambiguous_score_and_geometry | 161 | 18 |
| contour_postprocess | 145 | 125 |
| geometry | 3,280 | 2,759 |
| score_filtering | 75 | 52 |
| weak_response | 2,791 | 3,498 |

| Native LR2 height | Misses | Weak response at c0=0.30 |
|---|---:|---:|
| <4 px | 962 | 452 |
| 4–<6 px | 1,937 | 850 |
| 6–<8 px | 1,192 | 508 |
| 8–<16 px | 1,561 | 699 |
| ≥16 px | 802 | 282 |

At height ≥6 px, **1,489** of **3,555** valid-kernel misses have weak response.

Of those weak cases, **1,390** have B1-cal Clean coverage ≥0.50. Review the selected crops before deciding on PRC. This is a descriptive diagnosis on the validation used for calibration.
