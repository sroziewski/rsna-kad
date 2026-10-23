Subset: 481 studies (58 gold, seed-0 sample 400, audit 25); 5772 cells per table.

| table | model | rules | gold-58 AUC [CI] | parse% | verbatim% | rep/min | cell agr% | neg_unment | abs_vs_neg | subthr_neg | misattr | quote_miss | quote_fabr | audit-25 nu/sn/ma/qm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| awq_base | Qwen/Qwen3-32B-AWQ | base | 0.850 [0.820,0.880] | 100.0 | 96.6 | 38.4 | 100.0 | 127 | 260 | 178 | 105 | 141 | 25 | 11/6/4/7 |
| bf16_base | Qwen/Qwen3-32B | base | 0.847 [0.814,0.878] | 100.0 | 96.7 | 33.5 | 96.2 | 135 | 218 | 193 | 100 | 139 | 34 | 11/6/7/7 |
| fp8_base | Qwen/Qwen3-32B-FP8 | base | 0.845 [0.813,0.876] | 100.0 | 96.8 | 26.5 | 96.1 | 126 | 248 | 188 | 88 | 130 | 31 | 11/7/5/6 |
| awq_strict | Qwen/Qwen3-32B-AWQ | strict | 0.848 [0.819,0.877] | 100.0 | 97.7 | 31.4 | 94.5 | 104 | 368 | 189 | 88 | 90 | 21 | 7/6/2/2 |
| awq_strict2 | Qwen/Qwen3-32B-AWQ | strict2 | 0.846 [0.815,0.876] | 100.0 | 97.7 | 31.5 | 94.7 | 104 | 295 | 184 | 86 | 94 | 32 | 9/6/4/2 |
| fp8_strict | Qwen/Qwen3-32B-FP8 | strict | 0.846 [0.816,0.875] | 100.0 | 98.2 | 23.5 | 94.4 | 109 | 327 | 190 | 85 | 71 | 17 | 7/7/3/3 |

Per-label gold AUC:

| table | acl | mcl | med_men | lat_men | med_oa | lat_oa | pf_oa | effusion | synovitis | baker | contusion | fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| awq_base | 0.882 | 0.957 | 0.932 | 0.838 | 0.904 | 0.841 | 0.891 | 0.817 | 0.676 | 0.914 | 0.717 | 0.825 |
| bf16_base | 0.877 | 0.944 | 0.920 | 0.840 | 0.907 | 0.840 | 0.884 | 0.817 | 0.658 | 0.902 | 0.786 | 0.788 |
| fp8_base | 0.877 | 0.944 | 0.920 | 0.824 | 0.878 | 0.833 | 0.876 | 0.817 | 0.658 | 0.902 | 0.785 | 0.831 |
| awq_strict | 0.868 | 0.957 | 0.936 | 0.826 | 0.888 | 0.834 | 0.878 | 0.807 | 0.676 | 0.914 | 0.768 | 0.828 |
| awq_strict2 | 0.882 | 0.957 | 0.936 | 0.826 | 0.891 | 0.832 | 0.818 | 0.807 | 0.695 | 0.913 | 0.765 | 0.828 |
| fp8_strict | 0.868 | 0.944 | 0.911 | 0.826 | 0.905 | 0.843 | 0.868 | 0.839 | 0.676 | 0.914 | 0.731 | 0.828 |

Per-language gold AUC:

| table | en (n=28) | es (n=10) | tr (n=6) |
|---|---|---|---|
| awq_base | 0.862 | 0.873 | 0.728 |
| bf16_base | 0.855 | 0.873 | 0.687 |
| fp8_base | 0.852 | 0.873 | 0.687 |
| awq_strict | 0.854 | 0.859 | 0.753 |
| awq_strict2 | 0.855 | 0.848 | 0.753 |
| fp8_strict | 0.857 | 0.855 | 0.768 |

Paired gold-cell test (only the gold cells where the two tables disagree; `wins` = cell is closer to the binary gold label under the §2.1 state->score mapping):

| table | cells | mean|err| awq_base | mean|err| table | wins awq_base | wins table | best / worst labels (base-table) |
|---|---|---|---|---|---|---|
| bf16_base | 30 | 0.342 | 0.277 | 13 | 17 | contusion 1-8, fracture 4-2, pf_oa 3-0 |
| fp8_base | 34 | 0.273 | 0.332 | 21 | 13 | contusion 1-7, med_oa 4-2, pf_oa 4-0 |
| awq_strict | 57 | 0.250 | 0.319 | 37 | 20 | contusion 1-7, pf_oa 8-2, med_oa 11-0 |
| awq_strict2 | 52 | 0.229 | 0.337 | 35 | 17 | contusion 1-6, med_oa 9-1, pf_oa 11-2 |
| fp8_strict | 47 | 0.267 | 0.366 | 30 | 17 | contusion 3-6, med_oa 7-4, pf_oa 7-0 |

bf16_base vs awq_base: 96.2 % of cells identical; largest state changes (awq_base->bf16_base): absent->negated 101, negated->absent 43, affirmed->negated 25, negated->affirmed 14, affirmed->absent 13, hedged->affirmed 10, hedged->negated 6, absent->affirmed 6

fp8_base vs awq_base: 96.1 % of cells identical; largest state changes (awq_base->fp8_base): absent->negated 72, negated->absent 70, affirmed->negated 29, negated->affirmed 17, affirmed->absent 15, hedged->affirmed 10, absent->affirmed 7, hedged->negated 4

awq_strict vs awq_base: 94.5 % of cells identical; largest state changes (awq_base->awq_strict): negated->absent 181, affirmed->absent 34, absent->negated 33, affirmed->negated 29, hedged->affirmed 15, negated->affirmed 14, absent->affirmed 7, affirmed->hedged 3

awq_strict2 vs awq_base: 94.7 % of cells identical; largest state changes (awq_base->awq_strict2): negated->absent 135, absent->negated 62, affirmed->absent 34, affirmed->negated 29, negated->affirmed 13, absent->affirmed 10, affirmed->hedged 9, hedged->affirmed 8

fp8_strict vs awq_base: 94.4 % of cells identical; largest state changes (awq_base->fp8_strict): negated->absent 156, absent->negated 46, affirmed->absent 42, affirmed->negated 36, negated->affirmed 12, hedged->affirmed 11, absent->affirmed 9, hedged->negated 6
