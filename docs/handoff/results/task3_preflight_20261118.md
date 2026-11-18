# TASK 3 preflight — STOP / HPC unavailable

Audit date: 18/11/2026, Asia/Bangkok. Local source revision:
`0d0d2196cd03f58546e54be8954060e4d750ec70`.

**Overall status: FAIL at the HPC connectivity gate.** The user requested stopping
on FAIL. No TASK 3 inference, calibration sweep, PRC replay, training job or config
generation was executed. No TEST data was opened. Historical protocols, code and
result artifacts were not modified. This new report records the incomplete audit.

## Stop evidence

A read-only SSH audit was attempted against the configured `UIT-HPC` alias with
BatchMode, strict host-key checking, a 10-second connection timeout and 10-second
server-alive interval (maximum two unanswered probes). It returned exit code 1:

```text
Timeout, server hpc.uit.edu.vn not responding.
```

No remote audit output was received. The live checkout, installed MMOCR source,
current input hashes and runtime settings therefore remain unverified in this
audit. Previous successful jobs do not establish current readiness.

## Parameter and evidence table

PASS below means **verified in the committed/local source**, not a new HPC PASS.

| Item | Value / observation | Verification source | Status |
|---|---|---|---|
| HPC read-only connectivity | SSH timeout, exit 1; no remote output | Tool execution during this audit | **FAIL — STOP** |
| Historical output representation | `text_repr_type='quad'`, not `poly` | [Exported config, line 136](https://github.com/ncat218/MONKEY_V2/blob/0d0d2196cd03f58546e54be8954060e4d750ec70/docs/handoff/inference_config.py#L136); [runtime assertion](https://github.com/ncat218/MONKEY_V2/blob/0d0d2196cd03f58546e54be8954060e4d750ec70/rev4/stage_a.py#L202) | PASS source; differs from the TASK 3B premise |
| B0 native mask / score | `0.30 / 0.30` | `rev4/stage_a.py:202–204` | PASS source |
| B0-cal and B1-cal mask / score | Both selected `0.30 / 0.40` | [Calibration report](https://github.com/ncat218/MONKEY_V2/blob/0d0d2196cd03f58546e54be8954060e4d750ec70/docs/handoff/results/calibration_report.md) | PASS historical report; live receipts not rechecked |
| Historical `unclip_ratio` | `1.5` | `rev4/stage_a.py:202–204` | PASS source |
| Historical `min_text_width` | `5` | `rev4/stage_a.py:202–204` | PASS source |
| Historical `epsilon_ratio` | `0.01` | `rev4/stage_a.py:202–204` | PASS source |
| Historical `max_candidates` | `3000` | `rev4/stage_a.py:202–204` | PASS source |
| Historical DB shrink ratio | `0.4` | `rev4/stage_a.py:206–207` | PASS source |
| Installed MMOCR `poly`/`quad` parameter effects | Not inspected on the actual HPC installation | Requires live installed code and its hash | NOT VERIFIED |
| S1/S2 union | Expected 500 + 500 disjoint sources from historical split | [Handoff](https://github.com/ncat218/MONKEY_V2/blob/0d0d2196cd03f58546e54be8954060e4d750ec70/handoff.md), Rev.4 split rules | LIVE MEMBERSHIP/HASH CHECK NOT RUN |
| PRC valid-mask rate | Historical aggregate `608/20000 = 3.04%` | [PRC report](https://github.com/ncat218/MONKEY_V2/blob/0d0d2196cd03f58546e54be8954060e4d750ec70/docs/handoff/results/prc_report.md) | PASS historical report; per-filter rates NOT AVAILABLE from this audit |
| New inference, calibration or training job | None submitted | Execution history of this audit | PASS |
| Existing protocol/results | No edits | Local Git diff for this change contains only this new report | PASS |

### Local file fingerprints

These are fingerprints of the local evidence, not an assertion that current HPC
files match. The exported config has portable paths and is not byte-identical to
the historical absolute-path HPC config.

| File | SHA-256 |
|---|---|
| `rev4/stage_a.py` | `5aeaa225d8e6883a2373f5f67aa2fb88e2c7dd5203e2fa777ea4b7528f9ae153` |
| `docs/handoff/inference_config.py` | `0d1a3a0b301e75938d014e4c23d74ccbcaf02b85cc26d776ba58e9810f04c3b4` |
| `rev4/prc_plugins.py` | `d1f66779f8f063b30da07a9369d2bc0031c9835f11a8f8957831591e4e656b2f` |
| `rev4/prc_geometry.py` | `c544fbd47bccc9d72706d9ff5fe89233b00bf80e7b3ed71930267cf7ec6e14ba` |
| `rev4/calibration.py` | `9ca55b902ef0a29a6ca0d3458ffa8e92499167070dd72a2ec31de8301a563aeb` |

## Interpretation and scope corrections

1. **TASK 3A is a fixed-setting evaluation on previously inspected pilot sources.**
   S1 influenced the move from LR4 toward LR2; S2 was used to select LR2 through
   the registered gates. Although these sources were not used for the later
   checkpoint/threshold search, they are not a fresh, fully independent holdout
   of the entire research process. Report S1, S2 and their union separately and
   disclose this history. No new scores or paired intervals have been computed.
2. **Keep the historical `quad` path unless the task is explicitly changed.**
   TextOCR GT polygons and the evaluator's generic polygon geometry do not imply
   `DBPostprocessor(text_repr_type='poly')`. A switch from `quad` to `poly` would
   be an additional intervention and would invalidate a claim of unchanged
   inference settings. Actual installed-code effects of the proposed parameters
   must be audited before defining the sweep. No new grid has been selected.
3. **The old PRC aggregate does not reveal the mask rejection cascade.**
   Per-filter survival rates and causal attribution have not been established.
   If a diagnostic replay is needed, label it as replay rather than retroactively
   claiming it was logged for every historical microbatch. The two ordinary
   Clean/LR2 samples in a training microbatch must not be confused with the
   paired Clean teacher view associated with its LR2 sample.
4. **An extended calibration is a separate exploratory result.** It must preserve
   the old `.30/.40` results, use an identical predeclared grid and selection
   rule for B0/B1, and remain confined to remaining-validation. Do not choose its
   grid using any new S1/S2 results. No such calibration was run in this audit.

## Task status

| Task | Status | Missing evidence / next prerequisite |
|---|---|---|
| 3A: B0/B0-cal/B1-cal/EDSR on S1 union S2 | NOT RUN after FAIL | Restore HPC access, verify locked source IDs/hashes/settings; preserve pilot-use disclosure |
| 3B: expanded calibration and IoU/score distributions | NOT RUN after FAIL | Audit actual `quad` code and hypothetical `poly` effects; predeclare the same grid for both models |
| 3C: PRC mask/augmentation/alignment diagnosis | NOT RUN after FAIL | Read actual run logs, transforms and masks; distinguish recorded evidence from replay |
| 3D: Clean-only and second-seed configs, GPU estimates | NOT PREPARED after FAIL | Resume audit before preparing configs; no training submission is authorized by TASK 3 |

**Resumption condition:** HPC access and live input/code verification must work.
Do not infer that an unavailable check passed. No experimental success/failure
claim can be made from this stop receipt.
