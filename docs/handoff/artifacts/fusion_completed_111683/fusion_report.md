# Fusion and upsampling study

One allocation: technical probe and continuous 1000-update pilot gate, then 10000 total updates per arm.
Backbone frozen; FPNC and DBHead trainable in every arm. Fixed mask .40 / score .60 primary. No TEST.

| Model | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP | Tiny LR2 recall |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 79.43% | 76.90% | 82.45% | 72.05% | 11997 | 45.79% |
| B-Clean | 80.86% | 75.51% | 88.66% | 65.75% | 6578 | 15.96% |
| nearest | 79.53% | 76.94% | 82.42% | 72.14% | 12035 | 46.26% |
| asf | 79.44% | 76.95% | 82.32% | 72.24% | 12134 | 46.53% |
| bilinear | 79.12% | 76.69% | 82.29% | 71.81% | 12086 | 45.97% |
| dysample | 79.50% | 76.94% | 82.39% | 72.16% | 12065 | 46.22% |

## Fixed-threshold corruption transfer (same 500-source subset)

| Model | Clean | LR2 | LR4 | LR8 | Blur1 | Blur2 | JPEG30 | JPEG10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 79.41% | 77.47% | 54.14% | 7.80% | 66.06% | 22.73% | 76.79% | 58.91% |
| B-Clean | 80.46% | 74.81% | 40.51% | 7.03% | 53.05% | 12.96% | 76.24% | 58.88% |
| nearest | 79.38% | 77.49% | 53.83% | 7.75% | 65.99% | 22.50% | 76.70% | 59.07% |
| asf | 78.98% | 77.45% | 53.58% | 7.63% | 66.04% | 22.54% | 76.73% | 59.16% |
| bilinear | 78.74% | 77.18% | 53.27% | 7.80% | 65.42% | 22.45% | 76.35% | 58.56% |
| dysample | 79.29% | 77.44% | 53.99% | 7.80% | 65.98% | 22.61% | 76.76% | 59.16% |

## Paired source Hmean differences

- asf_minus_nearest / clean: -0.098 pp, 95% CI [-0.22672988156919344, 0.015834304339601403].
- asf_minus_nearest / lr2: +0.014 pp, 95% CI [-0.07860044722349274, 0.11322890149111835].
- bilinear_minus_nearest / clean: -0.418 pp, 95% CI [-0.5512128323285472, -0.27655181581677624].
- bilinear_minus_nearest / lr2: -0.244 pp, 95% CI [-0.3634298336573069, -0.12530866454152806].
- dysample_minus_bilinear / clean: +0.379 pp, 95% CI [0.25012687638184733, 0.5084741590923583].
- dysample_minus_bilinear / lr2: +0.246 pp, 95% CI [0.12572572365679185, 0.36436093898635663].
- dysample_minus_nearest / clean: -0.039 pp, 95% CI [-0.1119844935041911, 0.03591526822568326].
- dysample_minus_nearest / lr2: +0.002 pp, 95% CI [-0.07206221443543909, 0.07761347770534721].
- nearest_minus_B1 / clean: +0.103 pp, 95% CI [-0.019255100250558654, 0.22767189334999866].
- nearest_minus_B1 / lr2: +0.036 pp, 95% CI [-0.06776971677211391, 0.1532398255953557].
- asf_minus_B1 / clean: +0.005 pp, 95% CI [-0.18910915882392548, 0.1833631494352272].
- asf_minus_B1 / lr2: +0.051 pp, 95% CI [-0.07677518578957765, 0.18403030659282976].
- bilinear_minus_B1 / clean: -0.315 pp, 95% CI [-0.49875996752118873, -0.12161401357417978].
- bilinear_minus_B1 / lr2: -0.208 pp, 95% CI [-0.3405504237975726, -0.06775471494739815].
- dysample_minus_B1 / clean: +0.064 pp, 95% CI [-0.06482236277621206, 0.19947762391780863].
- dysample_minus_B1 / lr2: +0.038 pp, 95% CI [-0.06218945842159457, 0.14898727938554623].

Threshold sensitivity uses the same declared settings on every condition; no target-specific best threshold is selected.
Bins use GT height mapped to native LR2 across conditions. See JSON for full counts, bins, threshold curves and costs.
This is one-seed exploratory development; source CIs do not capture seed or selection uncertainty. Robustness on corruptions does not prove transfer to a different dataset. Final TextVQA/ICDAR2015 TEST is not run.

Recovery of job106090: Nearest/ASF final checkpoints and verified prediction prefixes were reused.
Only temporary-file handling changed; recipe, split, evaluator and thresholds are unchanged. See recovery_import.json for exact parent hashes. Timing was remeasured in the recovery allocation.
