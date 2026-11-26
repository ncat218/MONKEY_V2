# EDSR x ASF v1 · frozen factorial inference

Requested26/11/2026. Pipeline: RGB input → fixed pretrained EDSR x2 → MonkeyOCRv2-AS → trained ASF-FPNC → DBHead → fixed quad postprocessing. No training or threshold selection.

## Weights and comparisons

- B1 is the completed10k fine-tuned detector, SHA `6df7a3fea16e1d3170af92212d060aab8040bb128fcfb63fb81eb7c8572ec58f`.
- ASF is the completed additional10k checkpoint from fusion111683, SHA `9cae7ad418cc906edb56667c03c0d5f9e25f02b7ad929e4024ff03ebaec38059`. Its full learned weights are loaded strictly with `FusionStudyDBNet`/`StudyFPNC(mode='asf')`; this is not B0 or an untrained ASF switch.
- EDSR is unchanged from [recheck163012](edsr-recheck-v1.md), SHA `2e407341a088033b3110edb33a1040891368aabcec25f010378417c1f01dbaf1`.

The combined report has6 cells: native/Bicubic/EDSR input for B1 and ASF. B1 cells are reused from job163012 only after it completes successfully; the new job computes the3 ASF cells. Slurm uses `afterok:163012` while the parent is pending/running. A failed parent blocks submission/execution; it is not silently treated as a valid control.

Main comparisons are EDSR+ASF versus B1, ASF alone, and EDSR+B1. Bicubic+ASF separates learned restoration from ordinary resizing. Source-paired interaction is measured as `(EDSR+ASF − ASF) − (EDSR+B1 − B1)` with a confidence interval. Gains from separate operating points or studies are not simply added.

## Fixed experiment

Same2,262 paired Clean/native-LR2 sources, same500-source LR4/LR8/blur/JPEG stress cohort, same Clean/LR2 anchors. All development was previously viewed. Main mask/score `.40/.60`, historical secondary `.30/.40`; quad,unclip1.5,width5,epsilon.01,max candidates3000 throughout. No TEST, new split, SSIM threshold, training or retuning.

Restoration is always-on, including Clean. Bins stay tied to original native-LR2 GT height. Every generated native/Bicubic/EDSR image must have exactly the same size and SHA as the corresponding B1 input from163012, otherwise stop. Native ASF must reproduce the completed fusion per-source counts/bins at both operating points. Official evaluator parity is checked on GPU for all8 conditions before full evaluation.

## Outputs and metrics

P/R/Hmean,TP/FP/FN,all4 size-bin recalls,Clean retention,matched-GT rescued/lost counts,2,000 source-paired bootstrap intervals and timing. Multiple endpoints are exploratory/unadjusted,one detector seed. Matched-ID transitions are not automatically visual causes. B1/ASF absolute timings come from different allocations and must not be called a controlled deployment-speed comparison.

Outputs: `.runtime/edsr_asf/v1/results/edsr_asf_report.md`, `edsr_asf_summary.json`, `complete.json`; durable resumable records per source/condition. Original baseline and fusion artifacts are preserved.

Run `python submit_edsr_asf.py --plan-only`, then `python submit_edsr_asf.py` from the HPC repo. One12h H200 MIG35GB allocation,8CPU,40GB RAM; a probe failure stops before full inference. Duplicate submission is guarded. Existing job163012 and its source files are left unchanged.
