# EDSR recheck v1 · frozen fine-tuned B1

Requested26/11/2026. Inference only; no new detector or SR training, threshold search, TEST access or split/protocol changes.

## Exact comparison

All branches use completed B1 job12187, selected10,000-update checkpoint, SHA `6df7a3fea16e1d3170af92212d060aab8040bb128fcfb63fb81eb7c8572ec58f`:

1. Native input → B1.
2. Bicubic x2 → same B1.
3. Frozen pretrained EDSR64 x2 → same B1.

EDSR weights SHA `2e407341a088033b3110edb33a1040891368aabcec25f010378417c1f01dbaf1`, same implementation and tile256/halo40 as the historical restoration run. Shared detector weights are never changed. Shapes are exactly2×actual input, including odd-sized LR2 sources; predictions map back using actual per-axis ratios. Node-local temporary PNGs and durable per-task records avoid the prior cleanup issue and permit checked inference resumption.

**Primary mask/score `.40/.60`; historical secondary `.30/.40`**. Both use quad,unclip1.5,min width5,epsilon.01,max candidates3000. No threshold selected separately for restoration or corruption. Calibration here changes postprocessing settings, not checkpoint weights or probability-network temperature scaling.

## Data and deployment

- Paired Clean/native LR2 on all2,262 remaining-validation sources.
- LR4/LR8,blur1/2,JPEG30/10 on the same500 hash-selected sources as fusion111683, with Clean/LR2 anchors.
- SR is always-on, including Clean. No ground-truth corruption oracle or untested LR router. Clean results expose possible degradation of already adequate images.
- All sources are already-used development. No new independent holdout or cross-dataset claim.

## Endpoints and gates

P/R/Hmean,TP/FP/FN,source-paired CI,recall in4 fixed native-LR2 GT height bins,Clean retention,rescued/lost matched GT and timing for all3 branches. Upscaling never changes the GT size-bin definition.2,000 paired source bootstrap draws,seed20261107. Multiple endpoints are exploratory and unadjusted; one B1 training seed.

Probe in the same Slurm job checks every condition, both fixed operating points and the unchanged official evaluator. Every native source must reproduce the corresponding frozen B1 counts/bins from completed fusion111683. Hash/identity/coverage mismatches stop the job. Historical common-point SR reproduction is reported separately; no overwriting old reports.

Timing includes SR read/restoration plus detector decode/forward/quad processing at the primary operating point. It excludes temporary PNG encoding,hash checks and metric evaluation; non-isolated timing is not production throughput. EDSR pretraining for bicubic degradation may favor these synthetic LR inputs; transfer tests do not remove that limitation.

## Execution

`python submit_edsr_recheck.py --plan-only` prints the lightweight plan. `python submit_edsr_recheck.py` submits one12h H200 MIG35GB job,8CPU,40GB RAM. Full audits/models run in the allocation. Completed B1 and EDSR weights are required; no download or training happens on the login node.

Outputs: `.runtime/edsr_recheck/v1/results/edsr_recheck_report.md`, `edsr_recheck_summary.json`, `complete.json`; task records under `records/<condition>/`. Each task stores3 branches and both operating points. The report includes both historical and current settings, not an oracle best setting per corruption.
