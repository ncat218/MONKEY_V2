# Locked TRAIN membership · 20/11/2026

[Download train_ids.csv](https://raw.githubusercontent.com/ncat218/MONKEY_V2/main/docs/handoff/data/train_ids.csv)
· [Verification receipt](train_ids_receipt.json)

This CSV contains the exact **15,222 TRAIN sources** used by B1 and B-Clean.
It was exported directly from the surviving, hash-verified TRAIN manifest,
preserving its row order. It is a source list, not the full annotation manifest.
It includes no images, LR2 files, validation sources or TEST data.

| Column | Meaning |
|---|---|
| `source_image_id` | Frozen source ID |
| `clean_relpath` | Image path relative to the user's TextVQA clean image root |
| `clean_sha256` | SHA-256 of the original clean image file bytes |

All 15,222 source IDs, relative paths and clean file hashes are unique. CSV uses
UTF-8 without BOM and LF line endings, with exactly the three columns above.

## Fingerprints

Original manifest:
`manifest__train__textvqa_clean_degraded_same_textocr_word_gt.jsonl`

```text
Bytes:   386129047
SHA-256: ca1b91636cbe025823041ce336ffb59affa826a073f666dd7ba386bcab6c983d
```

Published `train_ids.csv`:

```text
Bytes:   1765795
SHA-256: 8ac5cbd98db0ffcb67536c6629ed4839ecf8c8e82b927796d0f9208102805f60
```

These hashes identify different files; the CSV is not expected to have the
original manifest's hash. The receipt records both and the export checks.

## Use for the BN recalculation pilot

Join `clean_relpath` to the local clean image root. Verify each original image's
bytes against `clean_sha256` before conversion or resizing. No annotations are
required for the BN image pass. This source list does not choose or change the
BN recalculation recipe, Clean/LR2 proportions or evaluation/selection rules.

When native LR2 is required, use [the locked implementation](../../../rev4/training_data.py):
Pillow 11.3.0, RGB, bicubic to `max(1, W//2) × max(1, H//2)`, without upsampling
back to the clean canvas. B-Clean was trained on 100% Clean; B1 on 50/50
Clean/native LR2 from the same source set.

The original manifest name and historical degraded-image metadata do not imply
that B1 used the old compound degradations. This CSV shares only clean source
membership. No training, inference or Slurm job was run for this export, and no
historical data, splits, protocols or results were changed.
