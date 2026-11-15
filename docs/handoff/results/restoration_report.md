# LR2 restoration before B1-cal

2,262 paired remaining-validation sources.

| Input | Precision | Recall | Hmean | TP | FP | FN | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr2 | 77.62% | 73.10% | 75.29% | 57,178 | 16,487 | 21,043 | 136.9 |
| bicubic_x2 | 77.18% | 73.39% | 75.24% | 57,405 | 16,972 | 20,816 | 161.6 |
| edsr_x2 | 77.03% | 74.79% | 75.89% | 58,498 | 17,439 | 19,723 | 204.6 |

| Hmean difference | pp | Paired 95% CI (pp) |
|---|---:|---:|
| edsr_x2 − bicubic_x2 | +0.66 | [+0.27, +1.04] |
| edsr_x2 − lr2 | +0.60 | [+0.24, +0.96] |
| bicubic_x2 − lr2 | -0.05 | [-0.53, +0.29] |

| Native LR2 height | GT | LR2 recall | Bicubic recall | EDSR recall |
|---|---:|---:|---:|---:|
| <4 px | 5,158 | 45.02% | 45.35% | 48.86% |
| 4–<8 px | 29,878 | 72.78% | 73.09% | 74.87% |
| 8–<16 px | 26,711 | 78.99% | 79.10% | 80.21% |
| ≥16 px | 16,474 | 72.91% | 73.45% | 73.95% |

Latency includes restoration and detector decode, forward and postprocessing. Temporary PNG encoding, file integrity checks and evaluation are excluded.
