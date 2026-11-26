# Frozen Tri box-quality × ASF × EDSR

Requested26/11/2026. **Job166132 submitted, PENDING afterok163064;20/20 CPU tests PASS.**
[Submission receipt](handoff/artifacts/tri_quality_166132/). GPU probes are not yet run.
Parent163064 was already RUNNING at inspection, so it is preserved;
this is a separate inference-only successor with `afterok:163064`. No detector/scorer training,
cutoff selection, TEST access, protocol/split edits, or overwrite of earlier results.

## Frozen source

[Trí's repository](https://github.com/minhtridaoo/monkeyocrv2-mix/tree/7517ce53f86962e07fd440397f0c525faea45c00),
exact Git blobs for `mixab/__init__.py`, `boxquality_janek.py`, `boxquality_devA_janek.py`,
model and fit receipt. These files are attributed to Trí/Janek and left unchanged; only the
feature extraction and frozen prediction interfaces are used, not their fitting/CLI commands.
[Byte-level provenance](handoff/artifacts/tri_quality_frozen/provenance.json).

- Model SHA256: `81867a494e391db180d293635e63bfdd87ddac59b09b5aac9f18417159196eb9`.
- Fit receipt SHA256: `76386825b11f9fff7d31e6129f1d7cdde67897ce7633aadf205eb6552d5ab8c3`.
- 13 features, normalization and logistic weights unchanged. Candidate mask .40, unclip1.5,
  min width5, epsilon.01, quad, max3000, raw candidate floor .20.
- F raw cutoff .60; RAW control .5976406401908236; Q learned score cutoff .3315908613227198.
- Same TRAIN15222 IDs verified by CSV SHA256; no fitting or cutoff selection here.
- Historical CRLF source hashes differ from Git LF hashes only by line endings. Assets are
  stored with `-text` so the fit receipt retains the exact published bytes.

## Comparisons and data

18 cells = B1/ASF × native/bicubic×2/EDSR×2 × F/RAW/Q. Both detector checkpoints have
already completed fine-tuning; the scorer is transferred to ASF/restored inputs as-is.
This transfer may lose accuracy or tiny recall and is not assumed to be beneficial.

Same2262 Clean/LR2 paired sources and same500-source LR4/LR8/blur/JPEG cohort as the parent.
A1130/B1132 are reported descriptively; both are previously viewed development data.
No split is relabeled as a fresh holdout. Geometry, matching and native-LR2 GT height bins
remain fixed even for upscaled images. No GT edits for missing annotations.

Full differences: Q−F and Q−RAW for each input/detector; ASF+Q versus B1; EDSR+ASF+Q
versus ASF+Q, EDSR+B1+Q and EDSR+ASF. Paired interactions separate combined effects.
Report P/R/Hmean, TP/FP/FN, four size bins, matched-ID transitions, source-paired2000-bootstrap
CIs, and mean Clean/LR2 Hmean with the same source resamples across conditions.
One detector seed, repeated development use and unadjusted multiple endpoints remain limitations.

## Execution and stopping

One12h H200 MIG35GB allocation,8 CPUs,40GB RAM. Both-detector probes cover all8 conditions
and3 inputs before the full pass; instrumented polygons/raw/packed scores must be unchanged,
official and indexed matching must agree, and F controls must match parent per-source counts,
GT IDs and restored pixel bytes. Any failure stops. Partial records are resumable under the
same identity; a time limit is not a valid completed result.

Dedicated interleaved latency:20 fixed sources × Clean/LR2 ×3 inputs ×2 detectors,
five warm-up sources, two alternating F/Q passes. Preprocessing/forward/postprocessing/scoring
are included; restoration is measured once per input and added equally. PNG encoding,
hashing and metric evaluation are excluded. Report mean/median/p95; this shared-GPU audit
is not production end-to-end latency. Output hashes must match the evaluated predictions.

Launcher: `python submit_quality_combo.py --plan-only`, then `python submit_quality_combo.py`.
Output: `.runtime/tri_quality_combo/v1/results/quality_combo_report.md` and JSON/count hashes.
GPU/runtime checks run inside the allocation. Unit tests must pass before submission.
