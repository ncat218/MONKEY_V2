# Latent refinement: 10,000-update extension · 21/11/2026

The user requested a longer training comparison after the 1,000-update pilot
job 81918. The pilot showed near-zero adapter/control differences and did not
establish efficacy. This extension tests the same three arms at a larger budget;
it does not replace or relabel the pilot results.

| Arm | Trainable | Initialization | Optimizer updates |
|---|---|---|---:|
| A0 / head | DBHead | B1, fresh AdamW | 10,000 |
| A1 / plain | DBHead + plain residual adapter | B1, zero output adapter, fresh AdamW | 10,000 |
| A2 / gated | DBHead + gated residual adapter | B1, zero output adapter, fresh AdamW | 10,000 |

Backbone/FPNC remain frozen, including buffers. Same 15,222 TRAIN sources,
50/50 Clean/native LR2, sampler/augmentation, seed 42, microbatch 2, accumulation
2, FP32, DB detection loss, AdamW LR 2e-5, WD 1e-4, clip 5. Warmup is 100 optimizer
updates, followed by Poly .9 to LR×.01 at update 10,000. Save checkpoints at
2,500/5,000/7,500/10,000; evaluate only the final checkpoint, without searching.

The pilot checkpoint contains optimizer/scheduler states, but its Poly schedule
has already finished at LR 2e-7. Extending that state for 9,000 more updates is
not equivalent to a planned 10k schedule. All three new runs therefore start
afresh from B1 with identical schedules. Only the new update budgets are reported
as 10k; prior pilot/wasted GPU time remains part of total project cost.

## Gates and evaluation

- Preserve all pilot artifacts. Run under `.runtime/latent_refinement/long_10k_v1`.
- Verify the completed pilot report, model implementation, locked B1 checkpoint,
  TRAIN/validation/evaluator identities and audited H200 MIG runtime.
- Checked writes use flush/fsync/read-back. Repeat the GPU initialization,
  gradient and frozen-state probe before training; stop on FAIL.
- Hash all 20,000 microbatches' source IDs and augmented inputs, requiring equal
  streams across arms. Record learned module update norms and check Adam steps
  reach exactly 10,000. Track GPU peak before the logger can reset counters.
- Evaluate 2,262 remaining-validation sources in Clean/LR2, fixed .40/.60 primary
  and .30/.40 secondary. Same P/R/Hmean, FP/FN, four size bins, source-paired
  bootstrap (2,000, seed 20261107), and short matched timing screen.
- A1−A0 and A2−A0/A1 isolate adapter effects. Compare with unchanged B1/B-Clean;
  beating only a continued-head control is insufficient for a practical win.
- No LR4/blur additions, CRAFT/KD loss, threshold search, protocol edits or TEST.
  One seed and reused development remain limitations. Longer training may still
  yield a negative result; it is not evidence of efficacy in advance.

```bash
python submit_latent_long.py --plan-only
python submit_latent_long.py
```

One job runs the three arms sequentially on one H200 MIG 35 GB, 8 CPU, 40 GB RAM,
with a 6 h wall limit. The 1k runs took approximately 5 minutes training per arm;
linear extrapolation is about 50 minutes per 10k arm, plus probe/evaluation/queue.
This is an estimate, not a guaranteed runtime. Partial training is not silently
restarted; checkpoints are retained for an explicit recovery audit if needed.

Report: `.runtime/latent_refinement/long_10k_v1/results/latent_report.md`.
