# Fusion recovery job111683 · startup receipts

Source commit [727dcc2](https://github.com/ncat218/MONKEY_V2/commit/727dcc2), parent job106090. [Recovery design](../../../fusion-recovery-v2.md).

- `pre_submit_audit.json`: real parent identity/recipe, six cache/receipt hashes, both completed checkpoint SHA256 and exact evaluation prefixes PASS. No inference/training/TEST during this audit.
- `submission.json`: one Slurm job111683; a second launcher invocation detected it already queued/running and did not duplicate it.
- Verified startup: RUNNING on h200node02. Worker20/20 regression/MMOCR tests PASS, fresh GPU preflight PASS, corruption pixel probe PASS, and real recovery import PASS. Files `preflight.json`, `corruption_probe.json`, `recovery_import.json` record the checks. This is not a final performance report.

Results remain under `.runtime/fusion_study/recovery_v2` on HPC. Original `.runtime/fusion_study/v1` and both 10k checkpoints are preserved. Completed Nearest/ASF are reused; only missing ASF evaluations and unfinished Bilinear/DySample work remain. A full final report is written only after all gates and coverage checks pass.

The controller still logs `START train nearest/asf` when entering those phases; their completed receipts cause an immediate checked return before Runner construction. These markers do not mean another10k training run.
