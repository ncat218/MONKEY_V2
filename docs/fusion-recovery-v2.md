# Fusion recovery v2 · job111683 (parent106090)

**Job111683 submitted and RUNNING on h200node02 at the startup check.** Source fix commit `727dcc2`. [Audit PASS and submission receipt](handoff/artifacts/fusion_recovery_111683/). Full20/20 CPU/MMOCR tests passed on HPC and again in the worker. Fresh GPU preflight, frozen-corruption pixel check and actual parent-artifact import all PASS. This is a startup status, not a completed recovery report.

Job106090 FAILED after 03:30:42, exit1:0, during ASF evaluation: `TemporaryDirectory.cleanup → os.rmdir → ENOTEMPTY`. Nearest and ASF each completed 10,000 optimizer updates. Their final checkpoint SHA256 and full/partial prediction caches were checked before recovery. The error alone does not establish the underlying filesystem cause; the failed directory was empty when inspected later.

## Technical fix

Stress images and the corruption probe use node-local `SLURM_TMPDIR` (fallback `/tmp` on Linux), outside shared datastore scratch. Encoded RGB/PNG/JPEG bytes and detector preprocessing stay unchanged. Cleanup retries ENOTEMPTY/EBUSY at most four times with bounded backoff; permission errors and persistent failures still stop the job. Materialized PIL images are explicitly closed.

## Recovery and provenance

Run folder: `.runtime/fusion_study/recovery_v2`; parent `.runtime/fusion_study/v1` stays unchanged.

`python submit_fusion_study.py --recover --plan-only` shows the allocation; `python submit_fusion_study.py --recover` submits one job. A fresh technical/corruption probe must PASS before importing parent artifacts or training. The job then:

1. Checks the pinned parent identity/job106090, exact TRAIN/VAL/runtime identity, recipe, task ordering, six parent artifact hashes and both final checkpoint hashes. Frozen architecture, training, evaluator and source changes outside the declared technical whitelist are rejected.
2. Reuses B1/B-Clean/Nearest evaluation (7,524 tasks each), ASF evaluation prefix (5,700 tasks), and the two completed 10k training receipts/checkpoints. Imported identities are explicitly mapped into the new run with `recovery_import.json`; original files are preserved. Re-entry cannot overwrite an advanced prediction prefix with the older prefix.
3. Remeasures the short latency screen in the new allocation, resumes missing ASF evaluations, then runs Bilinear and DySample. Each new arm keeps its 1k technical pilot inside the same continuous 10k schedule; a FAIL stops subsequent phases. Nearest and ASF are not trained again.
4. Generates the complete six-model report only after all conditions/arms and stream checks pass. No TEST, protocol/threshold change or method selection occurs during recovery.

Existing Nearest/ASF training uses the same source/augmentation stream hashes. Fixed mask/score `.40/.60` remains primary; `.30/.40` remains secondary. Before the failure, ASF minus Nearest was +0.014 pp LR2 (source95%CI [-0.079,+0.113]) and -0.098 pp Clean. These are complete main-condition counts on development, not a completed stress-transfer study or evidence of an improvement.

## Verification

Regression tests cover transient/permanent cleanup errors, byte-identical corruptions at node-local paths, rejection of changed recipe/training/data/checkpoints, exact cache prefixes and height-bin consistency, idempotent import preserving resumed progress, completed-arm no-retrain behavior, and propagation of `--recover` to all Slurm child phases. Full MMOCR integration tests and the GPU technical probe run before unfinished training.
