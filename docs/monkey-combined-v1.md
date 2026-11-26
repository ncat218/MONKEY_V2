# One allocation: frozen quality scoring + research bundle

Requested26/11/2026: replace pending jobs165276 and166132 with one job. Running
EDSR+ASF163064 is preserved. The combined job depends on `afterok:163064`.

This changes scheduling only. Both original launchers, source hashes, recipes,
checkpoints, probes, split/evaluator rules and output roots stay unchanged.
The exact previously submitted research-bundle source is also published in Git.
No TEST is used. No new tuning or experiment arms are introduced by the merge.

Order in one GPU allocation:

1. [Frozen Tri scorer × ASF × EDSR](tri-quality-combo-v1.md): inference only, no refit.
2. [Research bundle](research-bundle-v1.md): guided zoom, seed43 B1/B-Clean, PRC2.

Children run in separate Python processes, releasing model and optimizer memory
between studies. A child technical failure stops the next study; completed records
and checkpoints stay in their original roots. Research's existing independent-phase
status/skip rules remain intact; the combined receipt requires its final flag to
be true and never reports partial research results as fully completed.

Time limit54h = original child caps12h+40h+2h orchestration margin, within QoS72h.
This is a maximum allocation limit, not an expected duration. One MIG35GB GPU,
8 CPUs,40GB RAM. Merging does not make the computations faster; it removes one
queued allocation and runs the studies sequentially.

Replacement procedure: verify both old jobs still PENDING; prepare/test code;
hold old jobs, submit the new job held, verify its dependency; cancel only the two
old pending jobs, preserve their original tickets in a replacement receipt, point
both study tickets to the combined allocation, then release the new job. Never
cancel a running job as part of this replacement. Keep the submitted source/launch
identities unchanged.

Launcher: `python submit_monkey_combined.py --plan-only --hold`, then
`python submit_monkey_combined.py --hold`. Releasing a replacement is an explicit
operational step after cancelling the old pending allocations, not done by the launcher.

Outputs:
- `.runtime/monkey_combined/v1/results/combined_report.md` (index and completion receipt).
- `.runtime/tri_quality_combo/v1/results/quality_combo_report.md` (full scorer study).
- `.runtime/research_bundle/v1/results/bundle_report.md` (full research study).
- Per-child stdout/stderr: `.runtime/monkey_combined/v1/logs/quality_JOB.log`
  and `research_JOB.log`; progress/count caches/checkpoints retain their original paths.
