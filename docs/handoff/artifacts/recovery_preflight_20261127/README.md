# Recovery preflight27/11/2026

Source commit94def7a. Nine recovery tests PASS on the original HPC Python/MMOCR stack,
using a temporary /tmp package because NFS writes are blocked by user quota.
GPU probes have NOT run. No new recovery job has been submitted.

Verified CPU regression: installed DBModuleLoss.get_targets mutates polygon coordinates
in-place: [40,40,80,40,80,60,40,60] -> [6,6,46,6,46,26,6,26]. With the same teacher
response, selected GT kernels fall1 ->0. The snapshot-head regression preserves the
teacher mask, keeps the student's existing target construction and yields gradients.

Remaining tests verify cropped EDSR coordinates, Q-retained boxes below raw .60 surviving
fusion, frozen raw-score ordering, candidate floor and nonfinite rejection, control parity,
no baseline retraining and explicit classification of the unchanged PRC2 low-active guard.

Deployment blocker: NFS fsync raises `OSError: [Errno122] Disk quota exceeded` even though
the shared filesystem has75TB available. The quota executable is unavailable on login01.
Repo .runtime occupies32GB; /datastore/dungmt/.cache/pip occupies1.1GB. Permission to clear
that shared-account download cache is pending. Datasets/checkpoints/results have not been removed.

Git fetch found two empty loose objects, a01ce602663ed9170d78c5d33e2050c06c80af2e and
9eb7b9776d2d177eb326c41601ec3c35e8baaf54. Exact objects have been verified from the local
published clone. The first empty object was quarantined under .runtime/git_object_repair_20261127;
writing its replacement failed at fsync due to quota. Object repair and pull remain unfinished.
No evidence establishes that these later Git failures caused job166501's earlier termination;
quota is a directly reproduced current blocker, and the recovery adds signal/phase diagnostics.

[Recovery specification](../../../monkey-recovery-v1.md).
