# PR/quad audit v1

This is an exploratory postprocessing and error-diagnosis extension for the
remaining-validation cohort. It does not alter the frozen Rev.4 protocol,
historical reports, checkpoints, or evaluator. It does not train and does not
read TEST. The cohort has already been used for calibration and model
selection, so outputs are descriptive and not an independent holdout result.

## Comparison

- Checkpoints: B0, B-Clean, and B1, with their existing locked weights.
- Conditions: paired Clean and native LR2 for all 2,262 remaining-validation
  sources.
- Existing common operating point: mask threshold 0.30 and score threshold
  0.40. The implementation first requires these counts to reproduce the
  historical EDA per image.
- Output geometry remains `quad`; the installed DBPostprocessor is pinned by
  SHA-256 `9bfff8aaaca103dd08d7565f2b7a5410ab20a6427fe1f0baad5a5d04d207c8b8`.

## Exploratory grid

Eight score thresholds are applied to each of nine geometry/mask settings:

| Setting | Mask | Unclip | Minimum width | Epsilon ratio |
|---|---:|---:|---:|---:|
| base | 0.30 | 1.5 | 5 | 0.010 |
| unclip 1.2 | 0.30 | 1.2 | 5 | 0.010 |
| unclip 1.8 | 0.30 | 1.8 | 5 | 0.010 |
| width 3 | 0.30 | 1.5 | 3 | 0.010 |
| width 7 | 0.30 | 1.5 | 7 | 0.010 |
| epsilon 0.005 | 0.30 | 1.5 | 5 | 0.005 |
| epsilon 0.020 | 0.30 | 1.5 | 5 | 0.020 |
| mask 0.20 | 0.20 | 1.5 | 5 | 0.010 |
| mask 0.40 | 0.40 | 1.5 | 5 | 0.010 |

Score thresholds: 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, and 0.70.
The same grid and rule is used for all three models. Per model, the exploratory
setting maximizes the unweighted mean of Clean and LR2 micro-Hmean; exact ties
prefer the original base 0.30/0.40 setting. This rule does not replace the
original calibration result.

## Audit and review

The public and installed quad implementation uses `epsilon_ratio` during
contour approximation, score filtering before unclip, `unclip_ratio` for
polygon expansion, and `min_text_width` after quad fitting. These are
postprocessing controls; they do not change the P-map or learned features.
The job hashes and checks the installed source before use.

Outputs include base-geometry PR/FP-versus-tiny-recall curves, the full
threshold frontier, counts through contour approximation/unclip/width checks,
and per-instance FP/GT diagnosis with a stratified sample of paired image and
P-map cards. Automatic taxonomy labels are proxies. Human review is required
before interpreting them as visual causes. See `human_review.csv` and leave
unreviewed labels blank.

Slurm launcher: `python submit_decision_audit.py --plan-only`, then
`python submit_decision_audit.py`. Results use the new versioned directory
`.runtime/rev4_decision_audit/v1/` and never overwrite historical results.
