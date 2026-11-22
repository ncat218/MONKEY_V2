# Latent refinement pilot

Same B1 init, 1000 optimizer updates per arm; frozen backbone/FPNC including buffers. DB detection loss only.
Primary fixed mask=.40 / score=.60; no threshold or checkpoint search. Development data only; no TEST.

| Model | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP |
|---|---:|---:|---:|---:|---:|
| head | 79.39% | 76.71% | 82.09% | 71.98% | 12282 |
| plain | 79.39% | 76.71% | 82.09% | 71.99% | 12282 |
| gated | 79.39% | 76.72% | 82.10% | 71.99% | 12279 |
| B-Clean | 80.86% | 75.51% | 88.66% | 65.75% | 6578 |
| B1 | 79.43% | 76.90% | 82.45% | 72.05% | 11997 |

## Paired source CI (2000 draws, seed 20261107)

- plain_minus_head, clean: +0.000 pp; 95% CI [-0.007822975743352889, 0.007535240493715396].
- plain_minus_head, lr2: +0.003 pp; 95% CI [-0.007053049963175184, 0.013665518563311104].
- gated_minus_plain, clean: -0.001 pp; 95% CI [-0.00816047959392302, 0.006602553073269678].
- gated_minus_plain, lr2: +0.005 pp; 95% CI [-0.005900461951314218, 0.016492428273793262].
- gated_minus_head, clean: -0.001 pp; 95% CI [-0.008404683485085107, 0.006341169558480842].
- gated_minus_head, lr2: +0.008 pp; 95% CI [-0.00053310931718481, 0.018742102665806885].
- gated_minus_B1, clean: -0.040 pp; 95% CI [-0.21848025837968993, 0.16453091312728416].
- gated_minus_B1, lr2: -0.185 pp; 95% CI [-0.3444961009905309, -0.034393711104552825].
- plain_minus_B1, clean: -0.039 pp; 95% CI [-0.2198956320246498, 0.16242392936154035].
- plain_minus_B1, lr2: -0.190 pp; 95% CI [-0.3554496427410275, -0.03815881704833082].

One seed and short pilot; source CIs do not quantify training-seed or selection uncertainty. No automatic winner or final TEST decision. See JSON for four GT height bins and training cost.
