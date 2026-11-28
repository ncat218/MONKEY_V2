# Exact ICDAR2015 reproduction cohort

500 image IDs in original annotation row order, image hashes and valid/ignored GT counts.
The CSV is an audit/index, **not a replacement for word-polygon ground truth**.

The actual annotation manifest is `textdet_test__recovered_byte_identical_to_Phase1.json`,
SHA256 `fd1a9d52dbd2b3ed86a16fef27cda35e4c44624d07c4f1c25338ea2ca660f718`.
Top-level `metainfo` and `data_list`; rows carry image path/dimensions and word instances
with polygon/bbox/label/ignore. Do not generate a new annotation converter for comparison.

HPC already has the annotation, `dataset_audit.json` and official image ZIP at:

`/datastore/dungmt/textvqa/Tuan/monkeyocr_workspace/ICDAR2015__Monkey_E0_Aprime_FP32_transfer/inputs/`

[Full data hashes, settings and evaluator specification](../../../icdar2015-transfer-v1.md).

On 28/11/2026 all 500 encoded image SHA256 values, source hashes, decoded dimensions
and EXIF orientations were verified against the historical dataset. There are 2,077
valid words and 3,153 ignored words; original-height valid bins <8/8-<16/16-<32/>=32
contain 0/137/826/1114 words. No eligible <8px word means N/A, not zero recall.

Job192199 reproduced historical B0 exactly: TP1800, FP178, FN277, Hmean88.7793%.
See the [completed transfer report](../../results/icdar2015_transfer_192199.md)
for adapted-model comparisons and the limits of transfer to original ICDAR2015.
