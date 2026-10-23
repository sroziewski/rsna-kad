# Szymon's gold-58 predictions (team-internal)

Same layout as Qile's files: StudyInstanceUID, set (= gold58), then the 12 findings. Gold-58 macro AUC:

| File | What | Gold-58 | LB |
|---|---|---:|---:|
| szymon_main_v5_C_gold58.csv | our own model: 6 ConvNeXtV2-Tiny 384, full-data seeds (gold studies held out of training targets) | 0.9253 | 0.948 |
| public_R7_base_gold58.csv | heliosli's public R7 notebook (Raptor CoAtNet 384 + compact ConvNeXt-S R7, 50/50) | 0.9356 | 0.956 |
| v3stack_LB0957_gold58.csv | per-finding rank blend 0.7 R7 + 0.3 ours (submission 57150980) | 0.9403 | 0.957 |

The R7 part is not bit-reproducible on GPU; neighbouring ranks may swap between runs.
Joint 0.5 v3stack + 0.5 Qile's blend: 0.9440 on gold-58 (both legs 0.940).
