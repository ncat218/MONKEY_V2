# FULL_R1 Low Resolution pilot

Validation sources: 500. B0 FP32. Paired bootstrap by source.

| Condition | Precision | Recall | Hmean | TP | FP | FN | Clean TP survival |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 77.47% | 65.12% | 70.76% | 11941 | 3473 | 6396 | N/A |
| lr2 | 77.44% | 48.94% | 59.98% | 8975 | 2614 | 9362 | 71.67% |
| lr4 | 65.69% | 17.33% | 27.43% | 3178 | 1660 | 15159 | 25.46% |
| lr8 | 28.96% | 3.15% | 5.69% | 578 | 1418 | 17759 | 4.38% |
| b_up_lr4 | 66.80% | 18.69% | 29.21% | 3427 | 1703 | 14910 | 27.38% |

LR4 Hmean drop 43.33 pp, 95% CI [37.08, 49.78].
B_up recovery ratio 0.041.

- G1 PASS
- G2 PASS
- G3 FAIL
- G4 manual review of B_up, maps, word sizes and examples

Review LR4 samples and P-maps before making an intervention decision.
