# Recovery166501 + fixed-window zoom/EDSR/ASF/Q

Requested27/11/2026. One36h MIG35GB allocation,8CPU,40GB RAM. Technical probes and full runs share
the allocation; any technical failure stops the next phase. No TEST, no split/evaluator edits,
no scorer refitting or new threshold sweep. All prior sources and reports remain unchanged.

## Reporting question and baselines

Primary question: after domain adaptation, can restoration, multi-scale fusion and quality scoring
improve native-LR2 word detection, especially tiny text? TextVQA images + TextOCR word polygons.

- B0: original detector checkpoint, establishes the starting point.
- B-Clean: same B0 init/TRAIN IDs/update budget as B1,100% Clean; isolates domain adaptation.
- B1:50% Clean/50% native LR2; tests added resolution exposure. Not a weak untrained baseline.
- ASF/EDSR/Q: component ablations on completed B1, with the continued-training control retained
  for interpreting ASF. Existing techniques integrated/evaluated by the group, not claimed novel.

Same2262 reused development sources, primary .40/.60:

| Model | Clean H | LR2 H | LR2 recall | LR2 tiny recall | LR2 FP |
|---|---:|---:|---:|---:|---:|
| B-Clean |80.86|75.51|65.75|15.96|6578|
| B1 |79.43|76.90|72.05|45.79|11997|
| B1+ASF |79.44|76.95|72.24|46.53|12134|
| B1+EDSR |79.07|77.79|74.05|49.75|12774|
| B1+ASF+EDSR |79.02|77.87|74.35|50.74|12986|
| B1+ASF+EDSR+Q |79.50|78.23|74.22|49.92|12132|

Full combination vs B1: LR2 H+1.334pp,95%source CI[1.003,1.680]; tiny+4.130pp[2.686,5.646].
These are descriptive development results from one method seed; source CIs are not seed CIs.
B-Clean comparisons must include precision/FP and uncertainty; B-Clean+EDSR remains unmeasured.
Reporting focus: problem/EDA -> controlled B-Clean/B1 -> EDSR/bicubic and ASF ablations -> Q tradeoff.
Clean retention is secondary; LR4/blur/JPEG, negative methods, cost and limitations remain visible.

## Recovery of completed work

Quality study is complete and is reused with its report/summary/record hashes, no repeated forwards.
Both seed43 baselines trained10000 updates; both5k/10k checkpoint hashes are verified.
Resume original B1/B-Clean evaluation (checkpoint choice5k/10k and original15-pair calibration);
completed10k B1 records and partial5k prefix are reused. Finish original B-Clean zoom.
The old research root remains historical; its PRC2 failure is never relabeled PASS.

Original combined job ended mid early-checkpoint evaluation with no final traceback. The cause of
the whole-job termination is not established. New orchestration adds per-phase return codes,
10s heartbeat and signal receipts, stderr-first best-effort error persistence and resumable phases.

## Verified PRC2 implementation bug and correction

Installed DBModuleLoss._get_target_single -> _generate_thr_map -> _draw_border_map modifies GT
polygon arrays in-place. Original PRC2 calls detection loss before reading GT for teacher kernels;
its preflight reads GT before that mutation. CPU reproduction on the installed stack:
polygon[40,40,80,40,80,60,40,60] becomes[6,6,46,6,46,26,6,26]; teacher selected1 ->0.

New `Rev4PRC2SnapshotDBHead` deep-copies the aligned LR2 sample before detection loss, leaving
the student loss and teacher coverage constants unchanged. It trains in a separate root and
does not resume the invalid200-microbatch old run. Same seed42,10000 updates, loss weight1,
teacher thresholds, sampling and active>=50% at200 guard. Explicit low-active guard outcome is
reported STOPPED_LOW_ACTIVE; technical errors remain failures. It is an implementation correction,
not proof that it caused the later whole-job termination or that corrected PRC will improve Hmean.

## Four new zoom arms

Z0=B1+zoom+EDSR; Z1=ASF+zoom+EDSR; Z2=Z0+Q; Z3=Z1+Q.
Use original B1-selected windows, maximum4, exactly the original replace-fusion rule. No GT window
selection. EDSR is applied only on crops here; the completed whole-image ASF+EDSR+Q is a paired control.
Same2262 Clean/LR2 sources; no new stress sweep in this extension. Native LR2 GT size bins fixed.

Frozen Q uses13 features, raw pool>=.20, q>=.3315908613227198. Features are extracted on each
originating global/crop P-map BEFORE coordinate conversion. EDSR x2 coordinates are divided by2,
then shifted by crop origin and mapped into Clean coordinates. Original raw scores order NMS.
Four arms share forwards; Q-selected low-raw boxes are not cut a second time at .60.

Probe must reproduce original Z0 counts/bins and completed global F/Q controls and official matching.
Both models/probe conditions PASS before resuming baseline evaluation and the full zoom factorial.
Report P/R/Hmean, FP/FN, four size bins and paired source CIs, alongside whole-image leader.
Timing is diagnostic/shared extraction and evaluation, not a deployment benchmark.

## Outputs and run order

1. New zoom technical probes.
2. Resume evaluation B1_s43 and BClean_s43; no baseline retraining.
3. Full zoom factorial, then original B-Clean zoom.
4. Corrected PRC2 training/evaluation (200-microbatch guard before the full budget).
5. New recovery report; old completed artifacts and failure evidence preserved.

Launcher `python submit_monkey_recovery.py --plan-only`, then `python submit_monkey_recovery.py`.
New root `.runtime/monkey_recovery/v1`; report `results/recovery_report.md`, new zoom report
`zoom_quality/results/report.md`; corrected PRC2 under `prc2_snapshot/`.
