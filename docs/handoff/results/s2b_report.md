# CS431 Rev.4 · S2-B descriptive

Sources 500. B0 FP32. Clean baseline reused from S2-A job 11907. Paired bootstrap 2000 replicates, seed 20261107.

| Condition | Precision | Recall | Hmean | TP | FP | FN | Clean TP survival |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 71.64% | 62.31% | 66.65% | 9865 | 3905 | 5967 | N/A |
| blur1 | 68.38% | 22.73% | 34.11% | 3598 | 1664 | 12234 | 34.60% |
| blur2 | 35.89% | 5.85% | 10.06% | 926 | 1654 | 14906 | 8.39% |
| jpeg30 | 75.60% | 50.89% | 60.83% | 8057 | 2601 | 7775 | 79.13% |
| jpeg10 | 73.35% | 35.07% | 47.46% | 5553 | 2018 | 10279 | 54.48% |

| Condition | Hmean drop pp (95% CI) | Clean TP → FN | G1 | G2 | G3 |
|---|---:|---:|---|---|---|
| blur1 | 32.54 [26.09, 38.32] | 6452/9865 (65.40%) | PASS | PASS | NOT ASSESSED |
| blur2 | 56.59 [50.55, 61.85] | 9037/9865 (91.61%) | PASS | PASS | NOT ASSESSED |
| jpeg30 | 5.82 [0.66, 9.84] | 2059/9865 (20.87%) | PASS | PASS | NOT ASSESSED |
| jpeg10 | 19.20 [13.13, 24.88] | 4491/9865 (45.52%) | PASS | PASS | NOT ASSESSED |

LR2 remains selected by the registered S2-A priority. τ = N/A.

B_unsharp and JPEG cheap recovery are N/A. Size bins, P-map diagnostics and failure taxonomy are in s2b_summary.json. See review_examples.json for image and P-map paths.
