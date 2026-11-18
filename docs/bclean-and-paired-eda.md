# B-Clean and paired adaptation EDA

This is a separate exploratory control and diagnostic branch. Historical Rev.4
protocol, training, checkpoints, predictions and reports remain frozen.
No TEST data are read. Native LR2 remains Pillow 11.3.0 bicubic downsampling only,
with floor-half dimensions and per-axis polygon restoration for evaluation.

## Question

How much of B1's improvement comes from ordinary TextOCR domain fine-tuning,
and how much is the incremental benefit of explicitly mixing native LR2?

| Item | B-Clean | Historical B1 |
|---|---|---|
| Initialization | Same B0 checkpoint | B0 |
| Train sources | Same frozen 15,222 IDs | 15,222 |
| Input views | 100% Clean | 50% Clean / 50% native LR2 |
| Trainable modules | Full backbone, FPNC, DBHead | Same |
| Optimizer updates | 10,000 | 10,000 |
| Microbatch / accumulation | 2 / 2 | 2 / 2 |
| Image presentations | 40,000 | 40,000 |
| Seed | 42 | 42 |
| Optimizer | AdamW, LR 2e-5, decay 1e-4, clip 5 | Same |
| Scheduler | 100-update warmup, PolyLR power .9 | Same |
| Augmentations | Original color, flip/rotate/scale, crop and resize/pad 640 | Same |
| Checkpoint selection | Best mean native Clean/LR2 Hmean at 5k or 10k; earlier tie | Same |
| Calibration | Same original 15 mask/score pairs, native-first tie | Same |

The original B1 sampler's exact source-index stream is reused. Each LR2 draw
becomes the Clean view of that same source, using its full-resolution annotation.
Two Clean draw groups preserve the original batch slots and number of draws;
they do not create new unique sources or double the optimization budget.
Random geometric augmentation is still present, so this controls the marginal
effect of explicit native LR2, not all exposure to small text or rescaling.
Equal draws do not imply identical transformed pixels or exact augmentation RNG
trajectories: crop behavior can depend on input geometry. One seed cannot measure
training variance or establish tiny differences.

## EDA

Inference covers only the 2,262 remaining-validation sources, both Clean and LR2.
This is development data already used for checkpoint selection, calibration and
method exploration, not an independent final benchmark. The diagnostic job does
not optimize any new thresholds.

Three operating points are reported: native 0.30/0.30, common 0.30/0.40, and each
model's selected original-grid calibration. Other DBPostprocessor parameters
remain unchanged, including the audited **quad** path.

Pairs: B0 → B1 initially; then B0 → B-Clean and B-Clean → B1. For every source,
condition and operating point, recover the frozen evaluator's maximum-cardinality
one-to-one matches and assert exact agreement with its GT assignments and FP
indices. New inference counts must reproduce the historical per-image grid counts
(or B-Clean's new saved grid) before any EDA is accepted. On FAIL, stop.

Artifacts include per-word rescued/regressed/persistent-TP/persistent-FN records,
assigned and best-candidate IoU/score, score margin, ambiguity flags, kernel
coverage, polygon P-map statistics, FP correspondences, word height/aspect/angle,
and actual native/resized/padded/backbone/FPNC/P-map dimensions.

Fixed word-height bins are <4, 4–<8, 8–<16 and ≥16 px on the **native LR2 AABB
grid**, also used to stratify Clean so populations are comparable. These are
diagnostic bins, not a replacement for frozen metric bins. Missing candidate
scores remain missing, never zero. FP correspondence uses IoU ≥0.5 only to describe
changes in outputs and does not alter detection matching. Geometry categories and
P-map proxies need image review before claiming a mechanism or root cause.

At common thresholds, choose up to two examples per pair/condition/transition/
height bin by fixed SHA256 ranking, including regressions. Export paired GT/pred
overlays and P-map crops. The 2,000-replicate paired source bootstrap uses seed
20261107 and micro metrics; intervals are descriptive after selection and do not
represent training-seed variability. EDA timings are not latency benchmarks.

## Execution and outputs

Launchers perform only lightweight planning/submission on the login node.
Frozen SHA256/runtime/split checks run inside the allocated H200 MIG worker
before training/inference. Each worker also runs the new synthetic regression
tests before starting the research workload. Launch tickets lock source hashes.
No historical artifact is rewritten. A partial training checkpoint causes a stop
for review, rather than silently spending a second full training budget.

```bash
cd /datastore/dungmt/textvqa/Tuan/MONKEY_V2
git pull --ff-only
python submit_paired_eda.py --plan-only
python submit_paired_eda.py
python submit_bclean.py --plan-only
python submit_bclean.py
# Replace the numeric job IDs by those actually submitted above:
python submit_paired_eda.py --include-bclean --after-job EDA_JOB_ID --after-job BCLEAN_JOB_ID
```

The last job has `afterok` dependencies, reuses the verified B0/B1 EDA caches,
and only adds B-Clean inference and three-model reports. No LR4 training,
adapter, automatic PRC changes or TEST benchmark is part of these jobs.

Runtime paths (relative to the HPC repo):

- `.runtime/bclean/v1/results/bclean_report.md` and `bclean_summary.json`
- `.runtime/paired_eda/v1/results/B0_B1/eda_report.md`
- `.runtime/paired_eda/v1/results/B0_BClean_B1/eda_report.md`
- Each EDA result folder: `model_metrics.csv`, `paired_transitions.csv`,
  `fp_analysis.csv`, `stratified_metrics.csv`, `failure_distributions.json`,
  `input_shapes.csv`, `selected_examples.json`, `examples/`, `eda_summary.json`,
  `mechanism_selection.md`, and `complete.json` after visualization finishes.
- Logs: `.runtime/bclean/v1/logs/` and `.runtime/paired_eda/v1/logs/`.

Resource estimate on the same H200 MIG 2g.35gb: historical B1 training alone took
5,332 s (~89 min). Allow approximately 2–3 GPU hours for B-Clean including
two native candidate validations and 15-pair calibration, with an 8-hour walltime
ceiling. Allow 1–3 hours for initial B0/B1 EDA and 0.5–2 hours for additional
B-Clean EDA, with a 6-hour walltime ceiling for each. EDA estimates have not been
measured and depend on dense polygon matching and shared filesystem throughput.
Each job requests 8 CPUs, 40 GB host RAM and one MIG GPU. Failed checks stop work.

## Interpretation before choosing a new method

Compare B0 → B-Clean (domain adaptation) and B-Clean → B1 (incremental mixed
resolution benefit), then inspect the exact rescued/regressed populations and FP
cost. Common-threshold comparisons separate weights from score calibration;
own-calibration comparisons reflect the same selection procedure for each model.
These controls guide a later mechanism choice. LR4 stress inference, a second seed
and independent confirmation remain separate follow-up experiments.
