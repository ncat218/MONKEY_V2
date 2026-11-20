# Independent latent and CRAFT trials · 20/11/2026

These are exploratory extensions of the main workflow, independent of Trí's
Mix A/Mix B. They consume the completed fixed-setting diagnosis job 73827 and
do not change historical protocols, splits or results. TEST is never opened.

## A: latent residual pilot

Three arms start from the same B1 checkpoint, with fresh AdamW optimizers:

| Arm | Adapter | Trainable |
|---|---|---|
| head / A0 | Identity | DBHead |
| plain / A1 | 256→96, depthwise 3×3, GELU, 96→256 residual | DBHead + adapter |
| gated / A2 | 256→128, depthwise 3×3, split into two 64-channel branches, GELU(U)×V, 64→256 residual | DBHead + adapter |

Both adapters use per-position channel normalization and zero output projection.
They preserve the single 256-channel FPNC tensor and spatial grid. Plain has
50,528 parameters; gated has 50,816. The gated block is a small adaptation of
[Restormer GDFN](https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py),
not a reproduction of the full image restoration model or a new general method.
External S1b weights are not used.

The backbone and FPNC are frozen, including running buffers, in every arm.
Each arm uses the same locked 15,222 TRAIN sources, 50/50 Clean/native LR2,
augmentations, seed 42, 1,000 optimizer updates, microbatch 2, accumulation 2,
AdamW LR 2e-5, weight decay 1e-4, clip 5. Warmup is 50 optimizer updates, then
Poly .9 to LR×.01. The original DB detection loss is used without KD, character
supervision or an identity penalty. Only the final checkpoint is evaluated.

CPU tests check zero-init, parameter capacity, RNG isolation, guarded B1 loads,
gradient flow and frozen-buffer hashing. The allocation first runs real GPU
probes: maximum P-map difference ≤1e-6 from B1 at initialization; finite loss;
adapter output and body gradients; identical initial TRAIN draws/augmentation
hashes; frozen backbone/FPNC state unchanged. A failed probe stops before any
1,000-update arm. Probe models are discarded and training reinitializes B1.

Evaluation uses all 2,262 remaining-validation sources, paired Clean/LR2. Fixed
.40/.60 is primary; historical .30/.40 is also retained. No threshold search.
Reports compare plain−head, gated−plain/head, and the strong B1/B-Clean baselines,
with P/R/Hmean, FP/FN, four GT height bins, paired source CI, training time/VRAM.
A short within-phase timing screen compares B1/arm on eight fixed inputs, with
two warmups and five model+quad-decode repeats; it excludes image IO and
preprocessing and is not an isolated deployment benchmark.

## B: independent CRAFT screen and teacher-map QC

This stage runs the original General CRAFT detector, not a CRAFT auxiliary head
inside Monkey. Source is pinned at
`clovaai/CRAFT-pytorch@e332dd8b718e291f51b66ff8f9ef2c98ee4474c8`.
The MIT-licensed source is in `rev4/craft_vendor/`, with original/patched hashes.
Compatibility changes use relative imports and torchvision `weights=None`;
implicit ImageNet downloads are disabled. A reference preprocessing decoder is
disabled in favor of the project's audited image loading. See vendor provenance.

The original Google Drive link returned HTTP 404 during preparation. The same
named weight is distributed by EasyOCR at
[release pre-v1.1.6](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip),
as documented in [EasyOCR's config](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py).
The extracted file was verified against its published MD5
`2f8227d2def4037cdb3b34389dcf9ec1`; pinned SHA-256 is
`4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17`.
No claim of an author-published SHA-256 is made.

Settings are fixed official defaults: text threshold .7, low text .4, link .4,
canvas 1280, magnification 1.5, quad, no link refiner, FP32. No score calibration
or detector fusion occurs. Full heatmaps are decoded in FP32. Coordinates use
actual resize factors per axis and stride 2, then the same clean-coordinate
evaluator as Monkey. This corrects rounding differences from a nominal scalar
ratio; tests cover odd image dimensions. Component-maximum scores are clipped
to [0,1] for evaluator schema; these are not calibrated DBNet confidences.

On the same 2,262 validation sources, reports include standalone P/R/Hmean,
B1-TP/CRAFT-FN and B1-FN/CRAFT-TP, exclusive TP by size, and FP geometric
correspondence. These complementary GT sets are not a deployable fused result.
CRAFT uses different resizing and default settings versus calibrated B1; this
screen does not establish an architecture-only comparison or rule out a Clean
TRAIN teacher when LR2 detection is weak.

Teacher QC selects 200 TRAIN sources deterministically by seed/hash, verifies
TRAIN/validation disjointness, saves region/affinity maps and up to 32 word crop
cards. The frozen manifest has word polygons, not character GT/transcriptions;
word-level matching cannot certify character labels. Visual-QC maps are stored
in FP16 and are not training targets. Character review remains PENDING and no
auxiliary head is trained by this job.

## Launch and outputs

```bash
python submit_module_trials.py --kind latent --plan-only
python submit_module_trials.py --kind latent
python submit_module_trials.py --kind craft --plan-only
python submit_module_trials.py --kind craft
```

Both jobs wait for successful job 73827; if it has already completed, its
accounting state and complete receipt are checked instead of adding a stale
Slurm dependency. Full runtime, checkpoint and manifest audits happen in the
allocation. Any FAIL stops the phase. Historical source helpers stay unchanged.

Latent output: `.runtime/latent_refinement/pilot_v1/results/latent_report.md`.
CRAFT output: `.runtime/craft_screen/run_v1/results/craft_report.md`.
Latent requests one MIG 35 GB, 8 CPUs, 40 GB RAM, limit 6 h; CRAFT uses the same
resource profile with limit 3 h. These limits are not runtime estimates.

The primary metrics use development already examined repeatedly. Source-paired
CIs do not quantify one-seed training variance or selection uncertainty. A short
pilot does not automatically select a final method or authorize TEST evaluation.
