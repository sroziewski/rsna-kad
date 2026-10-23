# S3 error analysis (A+E+C)

## (a) Confident errors vs weak labels

Negatives 22,633, of which cFP 318 (1.405%); positives 7,650, cFN 195 (2.549%).

| label | neg | cFP rate | cFP among negated | cFP among absent | pos | cFN rate | cFN mild/hedged | cFN other |
|---|---|---|---|---|---|---|---|---|
| acl | 3152 | 0.002 | 0.001 (n=2698) | 0.006 (n=331) | 500 | 0.014 | 0.000 (n=13) | 0.014 (n=487) |
| mcl | 3870 | 0.025 | 0.015 (n=2961) | 0.014 (n=367) | 110 | 0.055 | nan (n=0) | 0.055 (n=110) |
| med_men | 1957 | 0.004 | 0.004 (n=1698) | 0.000 (n=218) | 1535 | 0.020 | 0.067 (n=15) | 0.019 (n=1520) |
| lat_men | 3126 | 0.010 | 0.010 (n=2736) | 0.015 (n=390) | 504 | 0.026 | 0.062 (n=16) | 0.025 (n=488) |
| med_oa | 1315 | 0.010 | 0.006 (n=1225) | nan (n=0) | 892 | 0.018 | nan (n=0) | 0.018 (n=892) |
| lat_oa | 1519 | 0.039 | 0.035 (n=1471) | nan (n=0) | 458 | 0.068 | nan (n=0) | 0.068 (n=458) |
| pf_oa | 1340 | 0.023 | 0.023 (n=1340) | nan (n=0) | 1199 | 0.028 | 0.060 (n=50) | 0.027 (n=1149) |
| effusion | 1673 | 0.008 | 0.004 (n=1282) | 0.023 (n=391) | 959 | 0.017 | 0.053 (n=19) | 0.016 (n=940) |
| synovitis | 60 | 0.017 | 0.017 (n=60) | nan (n=0) | 295 | 0.003 | 0.000 (n=13) | 0.004 (n=282) |
| baker | 1137 | 0.002 | 0.001 (n=1087) | nan (n=0) | 518 | 0.008 | 0.000 (n=41) | 0.008 (n=477) |
| contusion | 1690 | 0.008 | 0.008 (n=1690) | nan (n=0) | 470 | 0.053 | 0.105 (n=181) | 0.021 (n=289) |
| fracture | 1794 | 0.024 | 0.024 (n=1794) | nan (n=0) | 210 | 0.057 | 0.071 (n=98) | 0.045 (n=112) |

By voter agreement (all labels pooled):

| agree | neg | cFP rate | pos | cFN rate |
|---|---|---|---|---|
| all3 | 17,994 | 0.012 | 6,725 | 0.022 |
| llm2 | 3,458 | 0.019 | 887 | 0.052 |
| split | 1,181 | 0.036 | 38 | 0.000 |

By report language (pooled):

| lang | neg | cFP | pos | cFN |
|---|---|---|---|---|
| bg | 1,351 | 0.031 | 320 | 0.047 |
| de | 1,388 | 0.013 | 523 | 0.019 |
| el | 1,524 | 0.041 | 592 | 0.019 |
| en | 9,143 | 0.010 | 2,830 | 0.030 |
| es | 3,772 | 0.009 | 1,116 | 0.032 |
| fr | 413 | 0.017 | 174 | 0.017 |
| hr | 1,959 | 0.015 | 866 | 0.020 |
| nl | 636 | 0.016 | 279 | 0.007 |
| tr | 2,447 | 0.010 | 950 | 0.016 |

## (b) Gold-58 cells failed by every arm (E, A, A8, C and the blend)

10 of 696 gold cells (1.4%); 5 missed positives, 5 false alarms. The weak target points the same (wrong) way in 6 of them (label-driven), the opposite (right) way in 4 (image-driven: the label was right and every model still failed).

Per label: acl 0, mcl 0, med_men 0, lat_men 0, med_oa 0, lat_oa 4, pf_oa 1, effusion 0, synovitis 2, baker 0, contusion 2, fracture 1

Failure rate per arm (share > 0.5), all 696 cells: MVP 0.085, E 0.070, A 0.050, A8 0.047, C 0.049, A+E+C 0.042

| study (last 10) | lang | label | gold | weak target | LLM state/sev | V3 misrank share | label agrees w/ gold |
|---|---|---|---|---|---|---|---|
| 0568677187 | tr | contusion | 0 | 0.40 | affirmed/unspecified | 0.89 | yes |
| 0594649149 | es | pf_oa | 1 | 0.96 | affirmed/severe | 0.70 | yes |
| 1908452608 | tr | synovitis | 0 | 0.30 | absent/none | 0.70 | yes |
| 2853419550 | en | fracture | 1 | 0.09 | absent/none | 0.68 | NO |
| 3216559395 | en | lat_oa | 1 | 0.08 | absent/none | 0.72 | NO |
| 3432197492 | el | lat_oa | 0 | 0.29 | affirmed/mild | 0.73 | yes |
| 3451855003 | en | lat_oa | 1 | 0.05 | absent/none | 0.91 | NO |
| 6968676828 | hr | lat_oa | 0 | 0.52 | affirmed/moderate | 0.73 | NO |
| 8836556945 | es | synovitis | 1 | 0.29 | absent/none | 0.94 | NO |
| 8921482192 | en | contusion | 0 | 0.71 | affirmed/unspecified | 0.79 | NO |

Studies with >= 3 all-arm failures: 

## (c) Medial vs lateral

Discordant weak studies (exactly one side positive, the other negative, both cells kept). `AUC` = how well rank(med) - rank(lat) separates med+/lat- (1) from lat+/med- (0); 1.0 = sides never confused. `swap` = share of discordant studies where the WRONG side ranks higher. Bootstrap 1000.

| pair | group | n med+lat- | n lat+med- | AUC [CI95] | swap rate |
|---|---|---|---|---|---|
| med_men/lat_men | all | 960 | 181 | 0.978 [0.966, 0.988] | 0.153 |
| med_men/lat_men | side_est=L | 382 | 65 | 0.983 [0.967, 0.994] | 0.159 |
| med_men/lat_men | side_est=R | 377 | 79 | 0.967 [0.940, 0.988] | 0.175 |
| med_men/lat_men | side_est=unknown | 201 | 37 | 0.989 [0.975, 0.998] | 0.101 |
| med_men/lat_men | side_source=desc | 54 | 6 | 0.991 [0.965, 1.000] | 0.100 |
| med_men/lat_men | side_source=geometry | 253 | 48 | 0.966 [0.940, 0.987] | 0.140 |
| med_men/lat_men | side_source=tag | 450 | 89 | 0.979 [0.957, 0.995] | 0.191 |
| med_men/lat_men | side_source=unknown | 201 | 37 | 0.989 [0.976, 0.998] | 0.101 |
| med_oa/lat_oa | all | 162 | 54 | 0.878 [0.810, 0.934] | 0.204 |
| med_oa/lat_oa | side_est=L | 56 | 14 | 0.923 [0.849, 0.987] | 0.186 |
| med_oa/lat_oa | side_est=R | 69 | 18 | 0.855 [0.724, 0.959] | 0.218 |
| med_oa/lat_oa | side_est=unknown | 37 | 22 | 0.878 [0.764, 0.959] | 0.203 |
| med_oa/lat_oa | side_source=geometry | 29 | 5 | 0.945 [0.844, 1.000] | 0.235 |
| med_oa/lat_oa | side_source=tag | 91 | 25 | 0.870 [0.783, 0.949] | 0.207 |
| med_oa/lat_oa | side_source=unknown | 37 | 22 | 0.878 [0.773, 0.961] | 0.203 |
| control med_men/acl | all | 977 | 167 | 0.989 | 0.113 |
| gold med_men/lat_men | gold-58 | 14 | 11 | 0.961 | 0.080 (2/25) |
| gold med_oa/lat_oa | gold-58 | 9 | 5 | 0.867 | 0.286 (4/14) |
