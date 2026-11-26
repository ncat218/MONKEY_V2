# EDSR recheck163012 · verified startup26/11/2026

Submitted from code commit `b7c9c0d`. RUNNING on h200node01 at the latest startup check;50/7,524 tasks processed,each with native/bicubic/EDSR and both operating points. No training or TEST. This is not a final results report.

- `latest_job.json`: one inference-only Slurm job163012.
- `preflight.json`: GPU smoke PASS for all8 conditions,both frozen operating points; indexed matching agrees with the unchanged evaluator; native counts/bins reproduce completed B1.
- Historical LR2 sentinel: native,bicubic_x2,edsr_x2 all reproduce old `.30/.40` counts and matched GT IDs on the probe source. Full historical reproduction remains to be reported after completion.
- Completed B1 detector SHA: `6df7a3fea16e1d3170af92212d060aab8040bb128fcfb63fb81eb7c8572ec58f`.
- EDSR SHA: `2e407341a088033b3110edb33a1040891368aabcec25f010378417c1f01dbaf1`.

[Protocol and execution](../../../edsr-recheck-v1.md). Main point `.40/.60`,secondary historical `.30/.40`; no retuning. Always-on SR includes Clean. Native-LR2 size bins stay fixed across branches. All development was previously viewed; one B1 seed; source intervals do not measure training-seed uncertainty.
