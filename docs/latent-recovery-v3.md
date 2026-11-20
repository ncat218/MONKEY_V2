# Latent pilot recovery v3 · 20/11/2026

Job 77412 failed after 42:14, exit 1. The GPU probe passed all three arms, with
zero initial P-map error, finite gradients and identical source/augmentation
draws. Head and plain each completed 1,000 optimizer updates; their final
checkpoints and training receipts survive. Head validation and timing are valid.

The plain validation/timing JSON files, gated start marker and later log files
are zero bytes. The actual exception was not preserved. Slurm reports about
7 GB peak host RAM versus 40 GB requested, without an OOM state. Shared storage
has free space and a current 1 MiB fsync/read-back probe passes; historical
quota or NFS failure is not established. Do not report a confirmed storage root
cause or conclude that the learning method failed.

## Recovery

- Preserve all v2 artifacts and use `.runtime/latent_refinement/pilot_v3`.
- Pin the failed parent job and both completed checkpoint hashes. Recheck the
  recipe, unchanged model/hook implementations, split, checkpoint and runtime.
- Reuse head/plain training and valid head evaluation, with explicit source
  receipts/hashes. The published metrics are not inspected to choose reuse.
- Recompute the damaged plain validation/timing from the same saved checkpoint.
- The gated arm has no recoverable checkpoint. Restart it from B1 under the
  original recipe. Its prior attempted update count cannot be established from
  the damaged logs; do not claim it never started or hide the wasted compute.
- Run the GPU probe again before any remaining training. No change to architecture,
  data, loss, optimizer, scheduler, seed, budget or fixed operating points.
- New recovery writes flush and fsync a unique temporary file, verify its bytes,
  rename it atomically, and verify again. Failed writes preserve the previous
  target. Failure traces go to stdout/stderr as well as a checked artifact.
- Frozen helpers/protocol/evaluator and old outputs are not modified. TEST is
  not opened; CRAFT is not resubmitted.

```bash
python submit_latent_recovery.py --plan-only
python submit_latent_recovery.py
```

Full parent/checkpoint/runtime audits run inside the allocation. Output report:
`.runtime/latent_refinement/pilot_v3/results/latent_report.md`.

The inherited training peak GPU counters are marked invalid because runner
counter resets may have affected them. Original elapsed times and checkpoint
provenance remain available; do not present their low counters as actual peaks.
