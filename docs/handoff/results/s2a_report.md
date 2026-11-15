# CS431 Rev.4 · S2-A

Validation sources 500. B0 FP32. Paired bootstrap 2000 replicates, seed 20261107.

| Condition | Precision | Recall | Hmean | TP | FP | FN | Clean TP survival |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 71.64% | 62.31% | 66.65% | 9865 | 3905 | 5967 | N/A |
| lr2 | 76.70% | 45.69% | 57.26% | 7233 | 2197 | 8599 | 70.16% |
| lr4 | 65.61% | 15.93% | 25.64% | 2522 | 1322 | 13310 | 24.09% |
| lr8 | 32.21% | 3.73% | 6.69% | 591 | 1244 | 15241 | 5.39% |
| b_up_lr2 | 77.36% | 46.48% | 58.07% | 7358 | 2153 | 8474 | 71.53% |

LR2 Hmean drop 9.39 pp, 95% CI [3.86, 13.68].
LR2 clean TP → FN 2944/9865 (29.84%).
LR2 lost GT with native height at least 6 px 57.88%.
B_up LR2 gain 0.80 pp, 95% CI [0.30, 1.29].

- G1 PASS
- G2 PASS
- G3 PASS

G4 qualitative review of B_up, maps and LR2 transitions.

LR2 selected. Blur/JPEG remains descriptive.
