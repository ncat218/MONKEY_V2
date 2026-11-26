# Completed fusion111683 and exploratory multi-metric analysis

Job111683 COMPLETED,02:35:53,exit0:0. Parent106090 failed03:30:42 at temporary-directory cleanup. Checkpoints and prefixes were recovered, then all six models completed7,524 paired tasks. All4 training arms have10,000 updates with the same source/augmentation stream. Final checkpoint SHA, report completion hash, per-condition coverage and primary aggregate counts were verified read-only on26/11/2026.

`fusion_report.md`, `fusion_summary.json`, `complete.json` are unchanged HPC outputs. No TEST or new inference during that verification. `multimetric_posthoc.json` is a separate added analysis of the existing predictions:2,000 source-paired bootstrap draws,seed20261107. These unadjusted intervals on repeatedly used development are exploratory, cover multiple endpoints and do not quantify training-seed variation. This analysis does not change the frozen primary Hmean endpoint or declare a final winner.

Every new fusion arm starts from the **completed B1 checkpoint**, SHA `6df7a3fea16e1d3170af92212d060aab8040bb128fcfb63fb81eb7c8572ec58f`. Shared weights are checked tensor-by-tensor before further training. AS backbone is frozen; FPNC, DBHead and new modules train10k additional updates. Nearest is the matched continued-training control. These runs do not start from unadapted B0.

ASF versus B1 on native LR2: tiny recall45.79→46.53%,delta+0.737pp,95%CI[+0.159,+1.207];4–8px recall+0.385pp,CI[+0.192,+0.610];overall recall+0.187pp,CI[+0.059,+0.329]. Hmean+0.051pp has CI crossing0. FP+137,CI[+9.975,+289];large-word recall≥16px-0.279pp,CI[-0.508,-0.050]. Thus the pipeline has measured small-text recall gains with tradeoffs, not simply no effect.

ASF versus matched Nearest: tiny recall+0.271pp,CI[-0.070,+0.569]. This leaves the contribution specific to ASF versus extra optimization unresolved. Original native-LR2 GT height bins stay fixed across inputs; no artificial gain from relabeling upscaled letters as larger text.

[EDSR recheck protocol](../../../edsr-recheck-v1.md) uses the same completed B1 and two frozen operating points, including Clean and corruption transfer. [Checkpoint index](../../checkpoints.md).
