# Recovery job189496: RUNNING,28/11/2026

Quota write probe passed again, Git objects were restored/verified, and code was
fast-forwarded to d819296. No shared pip cache was deleted. One recovery allocation
was submitted under the earlier request to finish job166501 and add the zoom factorial.

Nine CPU tests PASS. Actual GPU zoom probes PASS for B1 and ASF on Clean/LR2:
fixed original B1 windows, EDSR crop coordinates, unchanged F/Q global controls,
original B1 zoom+EDSR counts and official matching. The job has moved to resuming
the cached B1_s43 evaluation. Full new zoom and corrected PRC2 results are not yet available.

[Submission/probe receipt](submission.json), [recovery specification](../../../monkey-recovery-v1.md),
[complete comparison of available results](../../../results/comparison_20261128.md).

Checkpoint/data artifacts were preserved. The failed old PRC2 run remains failed;
the GT-snapshot correction trains in a new root with unchanged thresholds and guard.
