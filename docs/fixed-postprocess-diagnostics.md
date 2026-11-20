# Fixed .40/.60 residual diagnosis · 20/11/2026

This is the next diagnostic stage of the main MonkeyOCRv2-AS + FPNC + DBNet
workflow. It uses the two existing B-Clean and B1 checkpoints on the locked
2,262 remaining-validation sources, paired Clean/native LR2. Both models keep
the already selected postprocessor: mask 0.40, score 0.60, quad, unclip 1.5,
minimum width 5, epsilon 0.01, max candidates 3000. No threshold sweep, training,
TEST use, checkpoint selection, or protocol change occurs in this job.

The preceding [job 63985 bundle](handoff/artifacts/pr_quad_audit_v1/README.md)
contains the reference source counts at this setting. The new job validates
TP/GT/pred and tiny GT/TP for every image against those counts. On the same
forward it also reproduces the historical .30/.40 prediction exactly.

## Questions

- Which FP/FN and B-Clean-to-B1 rescues/regressions remain after stronger
  postprocessing?
- Which GTs were rescued or regressed by changing postprocessing within each
  checkpoint?
- Are the remaining candidates weak, below the score cutoff, or poorly aligned
  with GT? These are diagnostic proxies, not causal conclusions.
- Does visual inspection support a feature intervention after FPNC, a
  localization intervention, or further annotation review?

## P-map evidence

The legacy helper hardcodes the pixel cutoff at .30 and remains unchanged.
`fixed_diagnostic_core.kernel_at_threshold` uses the same valid ROI, ignored GT,
shrink geometry and rasterization, with an explicit cutoff .40. Its .30 mode is
tested against the unchanged helper. The weak-response proxy remains a kernel
coverage fraction below .30, now computed from P-map pixels strictly >.40.
Pixel threshold and coverage fraction are different quantities.

Per-image records use `fixed` and `common` as aliases for the same shared
`.40/.60` setting in this new cache. Historical cache aliases remain unchanged.
The taxonomy machinery reads the record's alias rather than assuming .30/.40.

## Review

Review cards compare the same source/FOV for B-Clean and B1. Each panel contains
a Clean reference, the actual native input with predictions, and a P-map.
Valid GT is green, ignored GT cyan, predictions orange and the selected target
magenta. The FP target is its clean-coordinate AABB, shared across both panels.
This context helps avoid treating unmatched text as confirmed background.

One deterministic minimum-hash case is selected per condition/group/height/score
stratum, with population/inclusion metadata. This is a diagnostic sample and
does not yield a precise prevalence estimate by itself. Human labels remain
blank/PENDING until reviewed. In particular, no automatic background-percentage
rule triggers adapter training.

## Execution and artifacts

Launcher: `python submit_fixed_diagnostics.py --plan-only`, then
`python submit_fixed_diagnostics.py` on UIT-HPC. The allocation requests one
`gpu:2g.35gb:1`, eight CPUs, 40 GB RAM and at most three hours. Time limit is not
an estimated runtime. Output is `.runtime/rev4_fixed_diagnostics/v1/` inside
MONKEY_V2. New data are checkpointed by image; review aggregation is checkpointed
in chunks of 64 tasks. Completed historical artifacts are only read.

Results include `fixed_diagnostics_report.md`, `fixed_diagnostics_summary.json`,
four GT height bins in `stratified_metrics.csv`, within-model
`postprocess_gt_transitions.csv`, source-paired bootstrap comparisons,
`failure_distributions.json`, and `review/` with CSV/JSON/pair cards.

The cohort has repeatedly been used for development. CIs are conditional on
the selected checkpoints/settings, with 2,000 paired source bootstrap draws,
seed 20261107. They do not quantify training-seed variation or make this a fresh
holdout. The result informs the subsequent architecture decision; it does not
automatically authorize or launch the adapter pilot.
