# Latent refinement · 10000 updates

Same B1 init, 10000 optimizer updates per arm; frozen backbone/FPNC including buffers. DB detection loss only.
Primary fixed mask=.40 / score=.60; no threshold or checkpoint search. Development data only; no TEST.

| Model | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP |
|---|---:|---:|---:|---:|---:|
| head | 79.47% | 76.87% | 82.40% | 72.04% | 12037 |
| plain | 79.48% | 76.86% | 82.39% | 72.03% | 12046 |
| gated | 79.47% | 76.86% | 82.39% | 72.03% | 12042 |
| B-Clean | 80.86% | 75.51% | 88.66% | 65.75% | 6578 |
| B1 | 79.43% | 76.90% | 82.45% | 72.05% | 11997 |

## Paired source CI (2000 draws, seed 20261107)

- plain_minus_head, clean: +0.009 pp; 95% CI [-0.002078491702257046, 0.020466374750719992].
- plain_minus_head, lr2: -0.013 pp; 95% CI [-0.02549780272581542, -0.0012995986487862783].
- gated_minus_plain, clean: -0.008 pp; 95% CI [-0.01916952357500251, 0.002764538973875074].
- gated_minus_plain, lr2: +0.006 pp; 95% CI [-0.003289782991451884, 0.017189061065964455].
- gated_minus_head, clean: +0.001 pp; 95% CI [-0.010611956116030608, 0.013028431585526405].
- gated_minus_head, lr2: -0.007 pp; 95% CI [-0.018453770955166038, 0.003666696739994251].
- gated_minus_B1, clean: +0.042 pp; 95% CI [-0.04085568854642868, 0.12749749116294226].
- gated_minus_B1, lr2: -0.037 pp; 95% CI [-0.11291224271451572, 0.05007439109743744].
- plain_minus_B1, clean: +0.050 pp; 95% CI [-0.03380604960615624, 0.13697256635134739].
- plain_minus_B1, lr2: -0.043 pp; 95% CI [-0.11968578809376074, 0.04122680782286147].

One seed; source CIs do not quantify training-seed or selection uncertainty. No automatic winner or final TEST decision. See JSON for four GT height bins and training cost.
