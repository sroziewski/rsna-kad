# S6 Kaggle runtime budget and efficiency score

A6000 bench, 1,300 studies: eff_v1 9.2 min, main_v2 63.3 min, main_v3 133.0 min (GPU s/study 0.21 / 2.53 / 5.61).
Kaggle/A6000 factor: MVP (prep-bound) 2.18; main_v2 (GPU-bound) <= 1.70 (submit->complete <= 107 min incl. queue).

main_v3 projected hidden runtime: 3.8 h (v2 factor, itself an upper bound) .. 4.8 h (MVP factor); NOT measured - read the real value off the Kaggle submission page.
Per-arm A6000 GPU-min (1,300 studies, z-TTA): A ~41, E ~14, C 67; on Kaggle x1.70: A ~1.16 h, C ~1.89 h.
- 9 h hard limit: room after main_v3 = 5.24 h -> 2.8 more C-sized arms or 4.5 A-sized arms (at the optimistic v3 estimate).
- 9 h with 25 % margin: room after main_v3 = 3.44 h -> 1.8 more C-sized arms or 3.0 A-sized arms (at the optimistic v3 estimate).
- §2.7 rule: projected x1.25 < 6 h: room after main_v3 = 1.04 h -> 0.6 more C-sized arms or 0.9 A-sized arms (at the optimistic v3 estimate).
- Dropping z-TTA on all 15 members halves GPU time: saves ~1.72 h; its AUC value has never been measured (needs one GPU OOF pass with/without the z-offset view).

## Efficiency track

score = -AUC/0.45 + runtime_s/32400 (lower is better; research.md §1, Base 0.5, maxAUC 0.95 - both assumed).
1 min of runtime = 0.00185 = 0.00083 AUC; 0.01 AUC = 12.0 min.

| entry | LB AUC | runtime (est.) | score |
|---|---|---|---|
| eff_v1 (E, 5 folds, K16) | 0.913 | 20 min | -1.9919 |
| arm A alone | 0.931 | 82 min | -1.9177 |
| main_v2 (A+E) | 0.929 | 107 min | -1.8658 |
| main_v3 (A+E+C) | 0.933 | 225 min | -1.6558 |

eff_v1 is prep-bound: 3 workers x 1.00 s/study = 7.2 A6000-min of its 9.2; GPU 4.6 min (overlapped). A GPU-free model would save at most ~4.3 Kaggle-min = 0.0035 AUC-equivalent.
Break-even AUC for a replacement at runtime t (same score as eff_v1 at LB 0.913):
- t = 5 min -> AUC >= 0.9005 (-0.0125)
- t = 10 min -> AUC >= 0.9047 (-0.0083)
- t = 15 min -> AUC >= 0.9089 (-0.0041)
- t = 20 min -> AUC >= 0.9130 (+0.0000)
- t = 30 min -> AUC >= 0.9214 (+0.0084)
- t = 45 min -> AUC >= 0.9339 (+0.0209)

C-type efficiency candidate (2 folds, K16, no TTA; est.): GPU 8.9 + prep 9.6 A6000-min -> ~11.8 A6000-min -> 20-26 Kaggle-min; break-even LB AUC 0.9131-0.9179.
