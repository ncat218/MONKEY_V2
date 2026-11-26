# Combined Monkey job166501

Submitted26/11/2026. Replaced jobs165276 and166132 after verifying both were still
PENDING and held; both old allocations are confirmed CANCELLED. Parent163064 is
preserved RUNNING. New166501 is PENDING with `afterok:163064(unfulfilled)`.

41/41 CPU tests PASS on the HPC stack. GPU probes remain PENDING inside the original
workers. [Submission and replacement receipt](submission.json).

One allocation runs [frozen scorer transfer](../../../tri-quality-combo-v1.md), then
the [original research bundle](../../../research-bundle-v1.md), using unchanged
child launch identities, recipes, checkpoints and output roots. [Orchestration](../../../monkey-combined-v1.md).

54h is the maximum allocation limit, not an expected runtime. No TEST is used.
The quality phase is inference-only; research retains the previously planned
seed43 B1/B-Clean training and PRC2, plus the zoom study.

Overall index: `.runtime/monkey_combined/v1/results/combined_report.md`.
Individual reports remain in their original roots recorded in the receipt.
