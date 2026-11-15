# Rev.4 M PRC validation

2,262 remaining-validation sources. Selected `iter_20000.pth` after 10,000 optimizer updates.

| Model | Mask | Score | Clean Hmean | LR2 Hmean |
|---|---:|---:|---:|---:|
| B1-cal | 0.30 | 0.40 | 77.55% | 75.29% |
| M native | 0.30 | 0.30 | 71.93% | 69.88% |
| M-cal | 0.30 | 0.40 | 77.57% | 75.27% |

On validation, M-cal differs from B1-cal by +0.02 pp on Clean and -0.02 pp on LR2.
PRC active in 608/20,000 microbatches.
