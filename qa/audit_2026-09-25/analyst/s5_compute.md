# S5 compute - training GPU-hours from train.log

| run | folds (finished) | epochs | GPU-busy h | log wall h | img/s | mean best-last weak (fold-level optimism) |
|---|---|---|---|---|---|---|
| main_convnext_s_384 | 5.0 (5) | 60 | 15.2 | 15.2 | 203 | +0.0079 |
| main_armc_convnextv2_t_384 | 5.0 (5) | 50 | 12.2 | 12.2 | 301 | +0.0057 |
| main_convnext_s_384_e8 | 5.0 (5) | 40 | 10.2 | 10.2 | 202 | +0.0036 |
| eff_effv2s_256 | 5.0 (5) | 40 | 4.4 | 4.4 | 465 | +0.0062 |
| mvp_effv2s_256 | 5.0 (0) | 30 | 3.4 | 3.4 | 467 | +nan |
| main_armc_v1_1_f0 | 1.0 (1) | 10 | 2.4 | 2.4 | 300 | +0.0046 |
| main_armA8_v2 | 1.0 (1) | 8 | 2.3 | 2.3 | 165 | +0.0020 |
| main_coatnet2_384 | 1.0 (0) | 3 | 1.8 | 2.3 | 54 | +nan |
| main_smoke_384_eager | 1.0 (1) | 2 | 0.2 | 0.2 | 48 | +0.0000 |
| main_smoke_384 | 1.0 (1) | 2 | 0.1 | 0.1 | 96 | -0.0000 |
| armc_sweep_c4 | 1.0 (1) | 2 | 0.1 | 0.1 | 112 | -0.0000 |
| armc_sweep_c1 | 1.0 (1) | 2 | 0.1 | 0.6 | 122 | +0.0000 |
| armd_sweep_d3 | 1.0 (1) | 2 | 0.1 | 0.1 | 148 | +0.0684 |
| smoke_effv2s_256 | 1.0 (0) | 4 | 0.1 | 0.1 | 458 | +nan |
| armc_sweep_c2 | 1.0 (1) | 2 | 0.1 | 0.1 | 142 | +0.1051 |
| armd_sweep_d1 | 1.0 (1) | 2 | 0.1 | 0.1 | 180 | -0.0000 |
| main_smoke_384_repro | 1.0 (1) | 2 | 0.1 | 0.1 | 96 | -0.0000 |
| armd_sweep_d1g | 1.0 (1) | 2 | 0.1 | 0.1 | 180 | -0.0000 |
| armc_sweep_c3 | 1.0 (1) | 2 | 0.1 | 0.1 | 180 | -0.0000 |
| armd_sweep_d2 | 1.0 (1) | 2 | 0.1 | 0.1 | 181 | -0.0000 |
| armd_sweep_d0 | 1.0 (1) | 2 | 0.1 | 0.1 | 131 | +0.0000 |
| main_armd_convnextv2_t_384 | 1.0 (0) | 0 | 0.0 | 0.1 | nan | +nan |
| pseudo_srcA | 1.0 (0) | 0 | 0.0 | 2.3 | nan | +nan |
| pseudo_srcB | 1.0 (0) | 0 | 0.0 | 2.3 | nan | +nan |

Total over all logged runs: GPU-busy 53.3 h, log wall 59.1 h (44 fold logs).

## Inference benchmarks (bench_infer.py, 1x A6000)

| bundle | s/study | prep s (per worker) | GPU s | 1,300 studies A6000 min |
|---|---|---|---|---|
| eff_v1 (E, K16, no TTA) | 0.423 | 0.99 | 0.21 | 9.2 |
| main_v2 (A+E, 10 members) | 2.983 | 1.39 | 2.55 | 64.6 |
| main_v2 same-day | 2.920 | 1.07 | 2.53 | 63.3 |
| main_v3 (A+E+C, 15 members) | 6.138 | 1.17 | 5.61 | 133.0 |
