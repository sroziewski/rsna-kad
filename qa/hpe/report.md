# HP-diverse seed ensemble of C_p (developer O, queued 2026-09-27 15:08)

Base recipe C_p = `configs/main_armc_pseudo.yaml` (convnextv2_tiny, 5 views, 384 px, targets v3_pseudo, 8 epochs,
seed 2026) = member 1. Four new members, each = C_p + a new seed + ONE perturbation declared before training:

| member | config / run_id | seed | perturbation |
|---|---|---|---|
| hpe2 | `configs/main_armc_p_hpe2.yaml` / `main_armc_p_hpe2` | 2126 | lr_backbone 1.0e-4 -> 7.0e-5 (x0.7) |
| hpe3 | `configs/main_armc_p_hpe3.yaml` / `main_armc_p_hpe3` | 2226 | lr_backbone 1.0e-4 -> 1.4e-4 (x1.4) |
| hpe4 | `configs/main_armc_p_hpe4.yaml` / `main_armc_p_hpe4` | 2326 | drop_path 0.1 -> 0.2 |
| hpe5 | `configs/main_armc_p_hpe5.yaml` / `main_armc_p_hpe5` | 2426 | epochs 8 -> 10 |

Seeds are 2026 + 100 n because train.py seeds each fold with `seed + fold`; consecutive seeds would give
member n fold k+1 the same torch seed as member n+1 fold k. folds_v1 unchanged, so every member has a full 5-fold OOF.

**Rule: all five members are averaged with equal weight; none is ever selected or dropped by score.** With
gold-58's paired noise of about +-0.009, picking the best of five on gold would be a winner's curse, i.e. selecting
the luckiest draw rather than the best recipe.

Queue: `hpe_gpu0` waits for the exact screen `armn_full`, then hpe2 -> hpe4. `hpe_gpu1` waits for the exact screen
`xfmr_q`, then hpe3 -> hpe5. Before each member the waiter also waits for >= 18 GB free on its GPU (C_p peak 14.3 GB)
because of co-tenants. Nothing waits on `hpe_*`. Chain: `qa/hpe/chain.sh`.

Judging (when all four land): paired gold-58 of (a) mean(C_p, hpe2..5) vs C_p, (b) main_v4 with its C_p slot
replaced by that mean vs main_v4. Weak/strict are contaminated for pseudo runs and are secondary. Plus a Kaggle
runtime projection for 25 heavy members without z-TTA, against 9 h with a 25 % margin.
