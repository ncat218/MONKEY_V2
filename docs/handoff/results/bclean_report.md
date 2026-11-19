# B-Clean control: full fine-tuning on 100% Clean

Same B0 initialization and 10,000 optimizer updates as B1; same TRAIN IDs and source draw order.
Remaining-validation only. Clean still receives the original random scale/crop augmentation.

| Model | Mask/score | Clean Hmean | LR2 Hmean |
|---|---|---:|---:|
| B0-cal | 0.30/0.40 | 73.30% | 61.95% |
| B-Clean native | 0.30/0.30 | 73.79% | 69.46% |
| B-Clean calibrated | 0.30/0.40 | 78.79% | 74.36% |
| B1-cal | 0.30/0.40 | 77.55% | 75.29% |

Paired instance transitions and source-bootstrap differences are produced by submit_paired_eda.py --include-bclean.
One training seed. These validation values support model/threshold selection; they are not final test results.
