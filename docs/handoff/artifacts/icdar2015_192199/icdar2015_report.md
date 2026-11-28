# Original ICDAR2015: frozen pipeline transfer

500 original images; 2,077 valid GT and 3,153 ignored GT. Byte-identical Phase1 manifest/archive verified.
B0 trained on ICDAR TRAIN; test previously viewed during reproduction. This is transfer back, not fresh zero-shot.
Official installed MMOCR vanilla matching: IoU >0.5; prediction ignore overlap >0.5. No TextOCR matcher.
No training, checkpoint choice, threshold search or method selection on this test. EDSR always on where specified.
Tiny = original GT y-extent <8px; small additionally reported as <16px. Bins fixed before SR. N/A if no eligible GT.

| Pipeline | P % | R % | H % | Tiny <8 R % | Small <16 R % | TP | FP | FN | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0/reproduce | 91.00 | 86.66 | 88.78 | N/A | 90.51 | 1800 | 178 | 277 | 181.1 |
| B0/frozen | 86.83 | 88.25 | 87.54 | N/A | 91.24 | 1833 | 278 | 244 | 181.0 |
| B0/common | 88.00 | 86.13 | 87.06 | N/A | 90.51 | 1789 | 244 | 288 | 180.8 |
| B-Clean/native/F | 77.52 | 87.48 | 82.20 | N/A | 92.70 | 1817 | 527 | 260 | 178.0 |
| B1/native/F | 67.10 | 87.10 | 75.80 | N/A | 90.51 | 1809 | 887 | 268 | 177.7 |
| B1/bicubic_x2/F | 67.96 | 86.71 | 76.20 | N/A | 89.05 | 1801 | 849 | 276 | 284.9 |
| B1/edsr_x2/F | 72.12 | 85.80 | 78.36 | N/A | 88.32 | 1782 | 689 | 295 | 496.6 |
| Nearest/native/F | 67.46 | 87.05 | 76.01 | N/A | 90.51 | 1808 | 872 | 269 | 178.2 |
| ASF/native/F | 67.42 | 87.19 | 76.04 | N/A | 89.78 | 1811 | 875 | 266 | 181.9 |
| ASF/bicubic_x2/F | 68.35 | 86.81 | 76.48 | N/A | 89.78 | 1803 | 835 | 274 | 288.1 |
| ASF/edsr_x2/F | 72.01 | 85.60 | 78.22 | N/A | 89.78 | 1778 | 691 | 299 | 500.7 |
| ASF/edsr_x2/Q | 71.76 | 85.65 | 78.09 | N/A | 89.78 | 1779 | 700 | 298 | 503.0 |

B0/reproduce: mask .30, epsilon .01, native score .30 then packed fixed .60; required TP/GT/pred=1800/2077/1978.
B0/frozen: TextOCR-selected .30/.45, epsilon .005. B0/common and other F cells: .40/.60, epsilon .01.
Q: frozen raw pool >=.20 and quality cutoff .3315908613227198; no re-fit or test cutoff selection.

## Paired Hmean changes versus B0 and B1

| Pipeline | vs B0 reproduce pp [95% CI] | vs B0 frozen pp [95% CI] | vs B1 pp [95% CI] |
|---|---:|---:|---:|
| B0/reproduce | +0.000 [+0.000, +0.000] | +1.243 [+0.543, +1.910] | +12.978 [+10.897, +15.083] |
| B0/frozen | -1.243 [-1.910, -0.543] | +0.000 [+0.000, +0.000] | +11.734 [+9.713, +13.872] |
| B0/common | -1.723 [-2.569, -0.915] | -0.480 [-1.373, +0.442] | +11.255 [+9.227, +13.394] |
| B-Clean/native/F | -6.581 [-8.068, -5.078] | -5.337 [-6.768, -3.888] | +6.397 [+4.794, +8.134] |
| B1/native/F | -12.978 [-15.083, -10.897] | -11.734 [-13.872, -9.713] | +0.000 [+0.000, +0.000] |
| B1/bicubic_x2/F | -12.579 [-14.728, -10.459] | -11.335 [-13.484, -9.257] | +0.399 [-0.127, +0.956] |
| B1/edsr_x2/F | -10.415 [-12.359, -8.556] | -9.172 [-11.177, -7.297] | +2.563 [+1.532, +3.652] |
| Nearest/native/F | -12.765 [-14.869, -10.693] | -11.522 [-13.688, -9.512] | +0.213 [-0.234, +0.644] |
| ASF/native/F | -12.735 [-14.896, -10.639] | -11.491 [-13.649, -9.480] | +0.243 [-0.336, +0.785] |
| ASF/bicubic_x2/F | -12.300 [-14.473, -10.232] | -11.056 [-13.195, -8.986] | +0.678 [+0.056, +1.309] |
| ASF/edsr_x2/F | -10.557 [-12.598, -8.666] | -9.313 [-11.266, -7.491] | +2.421 [+1.363, +3.552] |
| ASF/edsr_x2/Q | -10.684 [-12.679, -8.756] | -9.441 [-11.380, -7.596] | +2.293 [+1.257, +3.385] |

2,000 paired-image bootstrap replicates, seed20261107. CIs exclude training-seed uncertainty and historical test exposure.
Latency is a separate 20-source audit, three warmups, two repetitions, includes SR and deployed detector/postprocessing; shared GPU.
Q checks direct-F candidate parity during evaluation; the separate deployed timing excludes this diagnostic duplicate pass.
All per-image predictions/counts and all four bins retained in records/ and JSON. Negative transfer must be reported, not retuned.
