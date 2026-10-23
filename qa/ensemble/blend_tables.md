# Ensemble OOF analysis (targets v1)

4407 common studies (4349 weak, 58 gold); OOF-weak per §2.5 (binarised targets, macro over 12 labels), gold-58 vs `train.csv`. Bootstrap: 2000 replicates, seed 2026, rows resampled once and reused for every candidate, so the deltas are paired.

| candidate | OOF-weak | delta vs best single | 95% CI of delta | P(blend > best) | strict | gold-58 | gold CI95 |
|---|---|---|---|---|---|---|---|
| main_convnext_s_384 | 0.9215 | +0.0000 | [+0.0000, +0.0000] | 0.000 | 0.8524 | 0.8905 | [0.861, 0.919] |
| eff_effv2s_256 | 0.9080 | -0.0135 | [-0.0186, -0.0084] | 0.000 | 0.8388 | 0.8666 | [0.832, 0.897] |
| mvp_effv2s_256 | 0.8815 | -0.0400 | [-0.0467, -0.0334] | 0.000 | 0.8095 | 0.8466 | [0.813, 0.877] |
| mean(main_convnext_s_384 + eff_effv2s_256) | 0.9292 | +0.0077 | [+0.0051, +0.0103] | 1.000 | 0.8614 | 0.8964 | [0.864, 0.924] |
| rank(main_convnext_s_384 + eff_effv2s_256) | 0.9271 | +0.0056 | [+0.0029, +0.0084] | 1.000 | 0.8601 | 0.8964 | [0.864, 0.924] |
| mean(main_convnext_s_384 + eff_effv2s_256 + mvp_effv2s_256) | 0.9304 | +0.0089 | [+0.0055, +0.0126] | 1.000 | 0.8583 | 0.8973 | [0.867, 0.925] |
| rank(main_convnext_s_384 + eff_effv2s_256 + mvp_effv2s_256) | 0.9272 | +0.0057 | [+0.0024, +0.0093] | 1.000 | 0.8593 | 0.8918 | [0.862, 0.918] |

Best single run on OOF-weak: **main_convnext_s_384**.

## Paired bootstrap: P(row's OOF-weak > column's)

| | main_convnext_s_384 | eff_effv2s_256 | mvp_effv2s_256 | mean(main+eff) | rank(main+eff) | mean(main+eff+mvp) | rank(main+eff+mvp) |
|---|---|---|---|---|---|---|---|
| main_convnext_s_384 | - | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| eff_effv2s_256 | 0.000 | - | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mvp_effv2s_256 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 | 0.000 |
| mean(main+eff) | 1.000 | 1.000 | 1.000 | - | 1.000 | 0.139 | 0.966 |
| rank(main+eff) | 1.000 | 1.000 | 1.000 | 0.000 | - | 0.005 | 0.458 |
| mean(main+eff+mvp) | 1.000 | 1.000 | 1.000 | 0.862 | 0.995 | - | 1.000 |
| rank(main+eff+mvp) | 1.000 | 1.000 | 1.000 | 0.034 | 0.542 | 0.000 | - |

## Per-label OOF-weak AUC

| label | main_convnext_s_384 | eff_effv2s_256 | mvp_effv2s_256 | mean(main_convnext_s_384 + eff_effv2s_256) | rank(main_convnext_s_384 + eff_effv2s_256) | mean(main_convnext_s_384 + eff_effv2s_256 + mvp_effv2s_256) | rank(main_convnext_s_384 + eff_effv2s_256 + mvp_effv2s_256) |
|---|---|---|---|---|---|---|---|
| acl | 0.9650 | 0.9544 | 0.9048 | 0.9690 | 0.9677 | 0.9632 | 0.9606 |
| mcl | 0.8691 | 0.8494 | 0.7436 | 0.8819 | 0.8785 | 0.8712 | 0.8655 |
| med_men | 0.9589 | 0.9465 | 0.9177 | 0.9597 | 0.9594 | 0.9573 | 0.9557 |
| lat_men | 0.9081 | 0.8695 | 0.7666 | 0.9150 | 0.9064 | 0.9063 | 0.8922 |
| med_oa | 0.9354 | 0.9376 | 0.9194 | 0.9431 | 0.9439 | 0.9463 | 0.9466 |
| lat_oa | 0.8630 | 0.8535 | 0.8370 | 0.8688 | 0.8701 | 0.8753 | 0.8786 |
| pf_oa | 0.9112 | 0.8890 | 0.8614 | 0.9164 | 0.9151 | 0.9128 | 0.9108 |
| effusion | 0.9539 | 0.9457 | 0.9407 | 0.9573 | 0.9572 | 0.9581 | 0.9580 |
| synovitis | 0.9084 | 0.9470 | 0.9138 | 0.9459 | 0.9461 | 0.9501 | 0.9479 |
| baker | 0.9678 | 0.9598 | 0.9428 | 0.9717 | 0.9710 | 0.9687 | 0.9691 |
| contusion | 0.9165 | 0.8582 | 0.9147 | 0.9104 | 0.9039 | 0.9282 | 0.9200 |
| fracture | 0.9006 | 0.8860 | 0.9152 | 0.9108 | 0.9062 | 0.9273 | 0.9217 |

## Per-label gold-58 AUC

| label | main_convnext_s_384 | eff_effv2s_256 | mvp_effv2s_256 | mean(main_convnext_s_384 + eff_effv2s_256) | rank(main_convnext_s_384 + eff_effv2s_256) | mean(main_convnext_s_384 + eff_effv2s_256 + mvp_effv2s_256) | rank(main_convnext_s_384 + eff_effv2s_256 + mvp_effv2s_256) |
|---|---|---|---|---|---|---|---|
| acl | 0.9651 | 0.9632 | 0.9400 | 0.9779 | 0.9730 | 0.9779 | 0.9706 |
| mcl | 0.8912 | 0.8753 | 0.5760 | 0.8934 | 0.9184 | 0.8571 | 0.8141 |
| med_men | 0.9459 | 0.9219 | 0.8984 | 0.9435 | 0.9489 | 0.9495 | 0.9507 |
| lat_men | 0.8814 | 0.7596 | 0.7329 | 0.8559 | 0.8335 | 0.8447 | 0.8199 |
| med_oa | 0.9403 | 0.9721 | 0.9411 | 0.9659 | 0.9674 | 0.9705 | 0.9721 |
| lat_oa | 0.7901 | 0.7427 | 0.8259 | 0.7737 | 0.7814 | 0.7930 | 0.7969 |
| pf_oa | 0.8301 | 0.8185 | 0.8095 | 0.8443 | 0.8443 | 0.8366 | 0.8378 |
| effusion | 0.9429 | 0.9422 | 0.9646 | 0.9627 | 0.9640 | 0.9764 | 0.9789 |
| synovitis | 0.7921 | 0.7342 | 0.8118 | 0.8100 | 0.8065 | 0.8339 | 0.8256 |
| baker | 0.9484 | 0.9475 | 0.8931 | 0.9438 | 0.9511 | 0.9330 | 0.9348 |
| contusion | 0.8664 | 0.8354 | 0.8698 | 0.8677 | 0.8583 | 0.8731 | 0.8772 |
| fracture | 0.8917 | 0.8868 | 0.8965 | 0.9181 | 0.9097 | 0.9222 | 0.9236 |

## Diversity: OOF correlation main_convnext_s_384 vs eff_effv2s_256

| label | Pearson | Spearman |
|---|---|---|
| acl | 0.882 | 0.718 |
| mcl | 0.693 | 0.673 |
| med_men | 0.918 | 0.887 |
| lat_men | 0.787 | 0.693 |
| med_oa | 0.894 | 0.875 |
| lat_oa | 0.811 | 0.786 |
| pf_oa | 0.820 | 0.820 |
| effusion | 0.894 | 0.874 |
| synovitis | 0.786 | 0.737 |
| baker | 0.901 | 0.739 |
| contusion | 0.812 | 0.728 |
| fracture | 0.785 | 0.681 |
| **mean** | 0.832 | 0.767 |
