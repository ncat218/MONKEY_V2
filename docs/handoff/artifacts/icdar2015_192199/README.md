# ICDAR2015 original: completed job192199

28/11/2026, H200 h200node01, MIG2g.35gb. **COMPLETED, exit0:0, elapsed57m18s.**
Source evaluation code: commit [e30cfc0](https://github.com/ncat218/MONKEY_V2/commit/e30cfc0).
This directory preserves worker outputs and additional independent audits.
`PASS` denotes technical completion/input identity, not improved model quality.

**Outcome:** frozen ASF+EDSR+Q improves Hmean over B1 by2.293pp
[1.257,3.385], but trails B0 reproduce by10.684pp and B-Clean by4.104pp.
It does not increase recall on the 137 original-height<16px valid words versus B1.
No training or threshold selection on ICDAR test was performed.

- [Vietnamese comparison and interpretation](../../results/icdar2015_transfer_192199.md).
- `icdar2015_report.md`, `icdar2015_summary.json`: exact complete worker report/JSON;
  their SHA256 values are bound by `complete.json`.
- `source_counts.npz`: 500 image rows in original manifest order for each of12 cells.
  Keys replace `/` by `__`. Columns are TP,GT,counted-pred,
  then GT,TP for bins<8/8-<16/16-<32/>=32 original pixels.
- `component_contrasts.json`: additional predeclared component comparisons from those
  counts, 2,000 paired-image bootstrap, seed20261107; no inference/fit/tuning.
- `identity.json`, `latest_job.json`, `final_identity_audit.json`, `slurm_job.txt`:
  source/checkpoint/config/data/evaluator identity and runtime/scheduler evidence.
- `data_audit.json`: all500 image hashes equal, dimensions and EXIF PASS;
  2,077 valid GT,3,153 ignored. Valid height-bin GT denominators0/137/826/1114.
- `b0_reproduction_gate.json`: exact historical TP1800/GT2077/pred1978.
- `b0_per_image_metric_audit.json`: recomputed official matching from job11339 raw
  predictions; counts and matched GT IDs agree on **every one of500 images**.
  Scores/quads are not claimed bit-identical across the two inference runs.
- [500-source audit index](../../data/icdar2015_reproduce/source_ids.csv).

The dataset and old B0 raw predictions remain on HPC in
`/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/ICDAR2015__Monkey_E0_Aprime_FP32_transfer/`.
New raw per-image predictions remain in
`/datastore/dungmt/textvqa/Tuan/MONKEY_V2/.runtime/icdar2015_transfer/v1/records/`.
Images and GT polygons are not redistributed here. The source index is not GT.

The original 500-image ICDAR test was already viewed during reproduction and B0
was trained on ICDAR TRAIN. This is transfer back after TextOCR adaptation, not
fresh zero-shot evaluation. No method is selected again using these results.

Rebuild checks/CI/Vietnamese report:

```bash
python analysis/icdar2015_transfer_report.py
```
