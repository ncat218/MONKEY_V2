# EDSR+ASF163064 · queued26/11/2026

Submitted from source commit `b86589c`. Verified state PENDING(Dependency),`afterok:163012`; baseline163012 remains RUNNING and had1,050/7,524 tasks at that check. No results or GPU-probe completion claimed for the combined job yet.

12/12 CPU/MMOCR regression tests passed on HPC before submission. New worker code does not modify baseline163012 source or results. It uses the completed ASF checkpoint SHA `9cae7ad418cc906edb56667c03c0d5f9e25f02b7ad929e4024ff03ebaec38059` and the same frozen EDSR. No training/retuning/TEST.

B1/native,B1/bicubic,B1/EDSR controls are imported only from successfully completed163012, while3 ASF cells are evaluated in the new allocation. Source IDs,actual restored image bytes,checkpoint hashes,official matching and baseline counts/bins must agree before full inference. A failed dependency or probe stops this path.

[Design and comparisons](../../../edsr-asf-v1.md). Report location on HPC: `.runtime/edsr_asf/v1/results/edsr_asf_report.md`. Source-paired effects include combined versus B1,ASF,EDSR+B1 and an interaction term; no additive-gain assumption.
