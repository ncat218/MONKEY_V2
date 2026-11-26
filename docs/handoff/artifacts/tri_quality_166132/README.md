# Frozen Tri scorer combination: job166132

Submitted26/11/2026, source1362b30. Slurm verified `afterok:163064(unfulfilled)`;
job166132 PENDING. Parent163064 was RUNNING and was not cancelled or modified.

20/20 CPU tests PASS on the frozen HPC stack. GPU probes are PENDING and run inside
the allocation before the full pass; they have not been claimed as passed.

[Study specification](../../../tri-quality-combo-v1.md) and [submission receipt](submission.json).
18 cells: B1/ASF × native/bicubic/EDSR × F/RAW/frozen-quality scoring.
No training, cutoff selection or TEST. Report path is recorded in the receipt's output root,
under `results/quality_combo_report.md` after successful completion.
