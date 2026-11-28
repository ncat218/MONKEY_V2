# Original ICDAR2015: frozen transfer v1

Authorized 28/11/2026: evaluate the reporting pipelines on the exact 500 original
ICDAR2015 test images previously used for reproduction. No training, LR generation,
threshold tuning, checkpoint selection or modification of the historical protocol.

## Input identity: exact reproduction manifest

Existing HPC folder (verified present):

`/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/ICDAR2015__Monkey_E0_Aprime_FP32_transfer/inputs`

Required files:

| File | Identity |
|---|---|
| `textdet_test__recovered_byte_identical_to_Phase1.json` | SHA256 `fd1a9d52dbd2b3ed86a16fef27cda35e4c44624d07c4f1c25338ea2ca660f718` |
| `dataset_audit.json` | SHA256 `8aa000d1585b442eb6a7d329a3e566bc6c3cc09890e101e5ea01a3bab6a4bc07` |
| `ch4_test_images.zip` | SHA256 `ecfb2488333372b7381b14ccb5ac2127de88fb935f6759fe89e5c041b0e87358`; historical MD5 `97e4c1ddcf074ffcc75feff2b63c35dd` |

The annotation JSON is the MMOCR-format manifest, not the TextVQA training JSONL:
top-level `metainfo`, `data_list`; each row has `img_path`, `height`, `width`,
`instances`; each instance has `polygon`, `bbox`, `bbox_label`, `ignore`.
Paths inside the recovered JSON retain the historical `textdet_imgs/test/img_N.jpg`
structure. Evaluation resolves filenames in node-local staging; source JSON bytes
and row order remain unchanged. Audit JSON includes a SHA256 for each image.

Exactly 500 unique image IDs, 2,077 valid word polygons, 3,153 ignored word polygons.
Verify the ZIP, all 500 image hashes, annotation/audit hashes, dimensions and EXIF.
Do not recreate annotations from a different converter or automatically "fix" ignore
flags; any mismatch stops the evaluation.

## Frozen design before new predictions

Main seven reporting pipelines:

1. B0 at its TextOCR-selected frozen .30/.45, epsilon .005 setting.
2. B-Clean .40/.60.
3. B1 .40/.60.
4. B1 + ASF .40/.60.
5. B1 + EDSR x2 .40/.60.
6. B1 + ASF + EDSR x2 .40/.60.
7. B1 + ASF + EDSR x2 + frozen Q.

Additional controls: B0 common .40/.60, B1/ASF bicubic x2, Nearest continued-training
baseline. Separate historical B0 reproduction setting: native mask .30, epsilon .01,
postprocessor score .30 followed by packed score cutoff .6000000000000001.
**Before adapted models run, all 500 B0 historical results must sum to exactly
TP=1,800 / GT=2,077 / counted-pred=1,978 (FP178/FN277), Hmean88.7793%.**
This is the pinned FP32 reproduction check, not a claim that every upstream paper
number uses this fixed cutoff. Do not tune thresholds if it fails.

One forward can feed several fixed B0 postprocessors. No best-test-score sweep.
Weights are the already completed seed42 B0/B-Clean/B1/ASF/Nearest checkpoints,
not newly trained models. EDSR and Q are frozen, same hashes and implementation
as the TextOCR comparisons. EDSR always runs on the original RGB when included.
Q uses raw pool>=.20, cutoff .3315908613227198. No refit or new cut selection.

## Metrics and fairness

Use installed MMOCR `HmeanIOUMetric(strategy='vanilla')`: strict IoU>.5 and ignored-GT
intersection/prediction-area>.5. This differs from the TextOCR maximum-cardinality
matcher. No substitution of evaluator. Q filtering preserves original raw score
and polygon order, and the official metric does not reapply raw score .6 afterward.

P/R/Hmean, TP/FP/FN, GT-matched recall in original-image polygon y-height bins
<8,8-<16,16-<32,>=32 pixels. Report cumulative <16 recall as well. Size bins remain
fixed before bicubic/EDSR. A bin with no GT is N/A, not zero. Report bin denominators.
2,000 paired-image bootstrap replicates, seed20261107, including comparisons to
B0 historical, B0 frozen, B0 common, B-Clean, B1 and Nearest.

Separate latency audit: 20 source IDs chosen deterministically by hash, 3 warmups,
2 passes, end-to-end SR+deployed detector/postprocessing, excludes evaluator/hash/
temporary PNG encoding. Shared-GPU measurements are not production guarantees.
Q instrumentation validates direct F parity during evaluation. The separate
deployed latency pass excludes that diagnostic duplicate F postprocessing.

ICDAR2015 test has been viewed during reproduction and B0 was trained on ICDAR
TRAIN. This measures transfer back after adaptation on TextOCR, not fresh zero-shot.
Report negative transfer, FP and Clean/original-image costs as well as positives.
Do not change the selected method using these results.

## Execution and outputs

```bash
cd /datastore/dungmt/textvqa/Tuan/MONKEY_V2
python submit_icdar_transfer.py --verify-only
python submit_icdar_transfer.py --plan-only
python submit_icdar_transfer.py
```

One inference-only H200 MIG35GB allocation, 8 CPUs, 40GB RAM, 8h cap. No dependency
on the running zoom/PRC job; it does not change or cancel that job.
Checkpoint/data/evaluator/code hashes are frozen in `identity.json` before sbatch
and verified again inside allocation. CPU semantic tests run before inference.
The B0 exact-count gate stops the job before any adapted model if reproduction fails.
Every image record is written durably for resume; changes require a separate version.

Outputs under `.runtime/icdar2015_transfer/v1/`:

- `data_audit.json`, `identity.json`, `b0_reproduction_gate.json`.
- `records/`: predictions, original-image hashes, official counts and GT matches.
- `latency/`: dedicated per-path timings.
- `results/icdar2015_report.md`, `icdar2015_summary.json`, `source_counts.npz`, `complete.json`.
- `logs/` and explicit failure traceback; no fabricated results on failure.
