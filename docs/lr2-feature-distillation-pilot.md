# LR2 feature-distillation pilot v1

Prepared 18/11/2026. **Exploratory post-hoc extension** after reading Rev.4 validation,
PRC and EDSR results. This does not replace the frozen Rev.4 protocol or change
historical scores. HPC is under maintenance at preparation time; GPU readiness is
not yet established. The user will run the launcher when HPC returns.

Local validation: 19 CPU tests passed on Python 3.14/Torch 2.11, including the
new geometry, mask, gradient, equal-budget, failed-preflight and paired-summary
checks and existing PRC/calibration/diagnostic regression tests. Syntax compilation
and launcher help also passed. This is not an MMOCR/CUDA integration pass; the
worker's mandatory GPU probe remains pending on the historical Python 3.11 stack.

## Question and controls

Does paired Clean-to-LR2 feature supervision improve over equal-budget continued
fine-tuning of the strong B1 baseline? Inference retains the original detector.

| Arm | Initialization | Extra optimizer updates | Added loss |
|---|---|---:|---|
| Historical B1 | Job 12187 selected checkpoint | 0 | None, stored reference counts |
| continued | Same B1 weights, fresh optimizer | 1,000 | None |
| feature_kd | Same B1 weights, fresh optimizer | 1,000 | Masked normalized feature MSE |

Both new arms use the full frozen TRAIN frame of 15,222 sources and the same
balanced sampler, seed 42, data transforms, FP32, 50% Clean replay, microbatch 2,
accumulation 2, AdamW LR 2e-6, weight decay 1e-4, clip 5, 50-update linear warmup
and polynomial decay .9. A short pilot samples the beginning of the shuffled
training stream; it is not five epochs or a full repeat of the original B1 run.
Each arm uses only its final checkpoint, without checkpoint or LR sweeps.

The frozen teacher is B1 in eval/no_grad, on a Clean view with exactly the same
crop, flip, rotation and field of view as the LR2 view, at 2x spatial resolution
(up to 1280x1280 versus 640x640). It is warped directly from Clean, not reconstructed
from LR2. Clean teacher color augmentation is omitted, while the student retains
the B1 augmentation recipe. The fused FPNC tensor is the only distillation site.
Teacher features are area-downsampled to the student feature grid.

TRAIN word polygons define a foreground mask. Ignored and padded areas are
excluded, including feature cells that mix them. Fractional foreground area
weights preserve small text support at the feature grid. Channel-RMS-normalized
MSE is averaged over channels and weighted spatial area. Its fixed weight is
0.1, an exploratory implementation choice, not an optimized or registered Rev.4
value. Valid-mask rate, raw KD loss and GPU memory are recorded.

This differs from old PRC in teacher (B1), supervision site (features), scale
(2x teacher canvas) and mask selection (GT foreground without teacher matching).
The initial two-arm pilot measures the complete intervention. It cannot attribute
any gain to features alone. A positive signal motivates a same-teacher/same-mask
P-map control, then separate boundary-loss ablations, before stronger claims.

## Verification and evaluation

`--plan-only` verifies the existing hash chain and prints a plan, without sbatch
or inference. The allocated worker first runs synthetic alignment/mask/gradient
tests. Its GPU probe reproduces B1 counts on two paired validation sources,
then checks a real TRAIN microbatch for correct shapes, nonempty weights, finite
nonzero KD gradients, frozen teacher and unchanged student state_dict keys.
Failure stops the worker before training. Zero active masks in the first 100
training microbatches also stops the distillation arm.

After each arm, evaluate all 2,262 frozen remaining-validation sources in Clean
and LR2. Report both fixed historical pairs `.30/.30` and `.30/.40`; no fresh
threshold search in this pilot. Compute TP/FP/FN, micro P/R/Hmean and native-height
recall bins `<4`, `4-<8`, `8-<16`, `>=16` px. Paired-source bootstrap uses 2,000
replicates and seed 20261107. Compare both arms to historical B1 and feature_kd
directly to continued. These are descriptive development CIs, excluding
training-seed variation and model-selection effects. No TEST or S1/S2 tuning.

Training timing, active KD batches and GPU peak allocation are recorded. Evaluation
records rough native detector timing, excluding image integrity checks. It is not
a controlled latency study and cannot prove exactly zero latency overhead.
The teacher is not part of the student module tree, optimizer or checkpoint.

Review the effect size, uncertainty, Clean retention, FP and small-text recall
together. No new hard go/no-go threshold is imposed, and a pilot CI containing
zero alone does not justify claiming failure. Preserve negative results. A full
trial and any additional calibration require a separate versioned extension.

## Run after HPC maintenance

```bash
cd /datastore/dungmt/textvqa/Tuan/MONKEY_V2
git pull --ff-only
python submit_feature_pilot.py --plan-only
python submit_feature_pilot.py
```

Slurm: one H200 MIG `2g.35gb`, `mig2-35`, 8 CPUs, 40 GB RAM, six-hour limit.
No new packages or datasets are downloaded. Existing B1 prepared inputs, stack,
checkpoint and remaining-validation images are required.

Outputs stay under the checkout:
`.runtime/lr2_feature_distillation/pilot_v1/`.

- `identity.json` and `submission/latest_job.json`: recipe, hashes and job ID.
- `preflight.json`: actual GPU and data checks, only after they pass.
- `continued/` and `feature_kd/`: training checkpoints, costs and validation records.
- `results/pilot_report.md` and `results/pilot_summary.json`: paired results.
- `logs/job_<ID>.stdout.log` and `.stderr.log`: Slurm logs.

The launcher avoids duplicate active jobs. Finished arms and validation progress
can be reused only when identities match. An interrupted training arm is not
silently resumed or restarted; it stops for review to avoid unequal budgets.
