# CRAFT independent complementarity screen

Pinned General CRAFT, official default thresholds .7/.4/.4, canvas 1280, mag ratio 1.5, quad. No training or TEST.
B1 uses fixed .40/.60. Both use the unchanged TextOCR evaluator and actual per-axis coordinates.

| Model | Clean H | LR2 H | LR2 P | LR2 R | LR2 FP |
|---|---:|---:|---:|---:|---:|
| CRAFT | 70.35% | 47.00% | 86.14% | 32.32% | 4068 |
| B1 | 79.43% | 76.90% | 82.45% | 72.05% | 11997 |

## GT complementarity (separate one-to-one matching, not fusion)

- clean: {'both': 43842, 'b1_only': 17410, 'craft_only': 2345, 'neither': 14624}
- lr2: {'both': 24108, 'b1_only': 32252, 'craft_only': 1172, 'neither': 20689}

FP geometric correspondence (older=B1, newer=CRAFT; IoU >=.5, not a fused detector):
{'clean': {'persistent_fp': 1673, 'older_only_fp': 13080, 'newer_only_fp': 5219}, 'lr2': {'persistent_fp': 782, 'older_only_fp': 11215, 'newer_only_fp': 3286}}

TRAIN teacher-map QC: 200 sources, 32 cards, character review PENDING.
Word-level GT cannot establish correct character/affinity pseudo-labels. Teacher head training is not run.
Official CRAFT resizing and Monkey resizing differ; this screen does not isolate architecture effects. Low LR2 complementarity does not by itself reject a Clean TRAIN teacher. Source CIs are descriptive on repeatedly used development, not training-seed uncertainty.
CRAFT timing includes model and CPU map decode, excludes file IO/preprocessing; no direct claim versus historical B1 timings.
