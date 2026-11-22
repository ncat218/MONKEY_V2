# Fixed .40/.60 residual diagnostics

2,262 repeatedly used development sources; paired Clean/native LR2. No threshold sweep or training. TEST not read.
Both models use mask=0.40, score=0.60, quad, unclip=1.5, min width=5, epsilon=.01, FP32.
Every image reproduced the historical .30/.40 prediction and the selected sweep .40/.60 TP/GT/pred/tiny counts.

| Model | Condition | P | R | Hmean | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| B-Clean | clean | 83.01% | 78.82% | 80.86% | 61650 | 12620 | 16571 |
| B-Clean | lr2 | 88.66% | 65.75% | 75.51% | 51432 | 6578 | 26789 |
| B1 | clean | 80.59% | 78.31% | 79.43% | 61252 | 14753 | 16969 |
| B1 | lr2 | 82.45% | 72.05% | 76.90% | 56360 | 11997 | 21861 |

## B1 minus B-Clean

- clean: -1.426 pp; paired source 95% CI [-2.1137593492696696, -0.9027428975164732].
- lr2: +1.394 pp; paired source 95% CI [-1.2547357399224024, 3.719047047628214].

## Diagnosis and review

127 paired review cards; human review PENDING. Sampling is one deterministic minimum-hash case per condition/group/height/score stratum; inclusion probabilities and population weights are recorded. This small stratified sample is for diagnosis, not a precise prevalence estimate.
Unlike the historical review, all predictions/FP correspondence here use .40/.60. Kernel coverage uses P-map >0.40; weak-response proxy means <30% of valid kernel pixels pass this cutoff. These proxies are not demonstrated causes, and unmatched text is not automatically background.
Inspect review/examples with GT and P-map before choosing a feature or localization intervention. No adapter is automatically selected or trained by this job.

Artifacts: failure_distributions.json, stratified_metrics.csv, postprocess_gt_transitions.csv, paired_source_counts.npz, review/diagnosis_summary.json, review/human_review.csv and review/examples/.
CIs are conditional source bootstrap intervals after development selection, not training-seed variability.
