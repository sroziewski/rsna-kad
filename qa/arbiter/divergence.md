# Label divergence: weak labels vs gold-58, by cell type

4407 studies x 12 labels = 52,884 cells.

| cell type | cells | share |
|---|---|---|
| all three voters agree (`all3`) | 40,792 | 0.771 |
| LLM V1==V2, rule voter differs (`llm2`) | 7,782 | 0.147 |
| LLM variants disagree (`split`) | 4,310 | 0.081 |
| primary state absent on an under-reported key | 13,333 | 0.252 |
| evidence_ok and no mention_conflict | 47,664 | 0.901 |

Per label, share of cells with three-voter agreement and the state mix of those cells:

| label | all3 | llm2 | split | all3 affirmed | all3 negated | all3 absent |
|---|---|---|---|---|---|---|
| acl | 0.798 | 0.155 | 0.048 | 821 | 2,368 | 326 |
| mcl | 0.814 | 0.126 | 0.060 | 499 | 2,737 | 351 |
| med_men | 0.821 | 0.147 | 0.031 | 1,987 | 1,414 | 219 |
| lat_men | 0.786 | 0.177 | 0.037 | 850 | 2,229 | 384 |
| med_oa | 0.636 | 0.212 | 0.151 | 1,138 | 949 | 718 |
| lat_oa | 0.621 | 0.223 | 0.156 | 676 | 1,084 | 976 |
| pf_oa | 0.734 | 0.176 | 0.090 | 1,514 | 1,046 | 673 |
| effusion | 0.928 | 0.045 | 0.027 | 2,506 | 1,198 | 384 |
| synovitis | 0.965 | 0.023 | 0.012 | 481 | 20 | 3,752 |
| baker | 0.848 | 0.093 | 0.059 | 926 | 770 | 2,040 |
| contusion | 0.689 | 0.176 | 0.136 | 855 | 1,361 | 819 |
| fracture | 0.617 | 0.213 | 0.170 | 198 | 1,613 | 910 |

## 2. How often the weak label is right, per cell type (gold-58, 696 cells)

`weak claim` = state->score mapping of the primary voter binarised at 0.5 (affirmed/hedged positive, negated/absent negative).

| cell type | cells | weak claim == gold | claim pos rate | gold pos rate |
|---|---|---|---|---|
| all3 | 549 | 0.812 | 0.490 | 0.386 |
| llm2 | 94 | 0.745 | 0.330 | 0.202 |
| split | 53 | 0.736 | 0.283 | 0.170 |
| absent_under | 168 | 0.857 | 0.000 | 0.143 |
| evidence_ok | 639 | 0.790 | 0.484 | 0.362 |
| all | 696 | 0.797 | 0.453 | 0.345 |

## 3. Do the models reproduce the extractor's errors?

On the gold cells where the weak claim disagrees with gold, AUC of the model's prediction **against the weak claim**. 0.5 = the model ignores the extractor's mistake; above 0.5 = it reproduces it (and OOF-weak pays it for that).

| run | gold AUC (all 696 cells) | AUC vs weak claim on the 141 disagreeing cells [CI95] | AUC vs gold on the same cells |
|---|---|---|---|
| main_convnext_s_384 | 0.8905 | 0.5222 [0.310, 0.639] | 0.4778 |
| eff_effv2s_256 | 0.8666 | 0.4911 [0.334, 0.640] | 0.5089 |
| mvp_effv2s_256 | 0.8466 | 0.4750 [0.295, 0.652] | 0.5250 |
| blend(main+eff) | 0.8964 | 0.5270 [0.324, 0.663] | 0.4730 |

Cells: 141 of 696 gold cells have weak claim != gold. CI = 1000 bootstrap replicates over the 58 gold studies.

## 4. Where does the blend's OOF-weak gain come from?

OOF-weak AUC restricted to each cell type (macro over the labels that keep both classes there). `delta` = blend - arm A.

| cell type | kept cells | main_convnext_s_384 | eff_effv2s_256 | mvp_effv2s_256 | blend(main+eff) | delta blend-armA [CI95] |
|---|---|---|---|---|---|---|
| all | 30,283 | 0.9215 | 0.9080 | 0.8815 | 0.9292 | +0.0077 [+0.0051, +0.0101] |
| all3 | 24,719 | 0.9281 | 0.9162 | 0.8942 | 0.9369 | +0.0088 [+0.0055, +0.0122] |
| llm2 | 4,345 | 0.8808 | 0.8663 | 0.8158 | 0.8862 | +0.0054 [-0.0030, +0.0139] |
| split | 1,219 | 0.8443 | 0.8379 | 0.8367 | 0.8804 | +0.0361 [-0.0167, +0.0731] |
| absent_under | 5 | nan | nan | nan | nan | +nan [+nan, +nan] |
| evidence_ok | 29,993 | 0.9212 | 0.9076 | 0.8808 | 0.9287 | +0.0075 [+0.0048, +0.0100] |
| evidence_bad | 290 | 0.8837 | 0.7286 | 0.7324 | 0.8938 | +0.0102 [-0.0175, +0.0496] |

