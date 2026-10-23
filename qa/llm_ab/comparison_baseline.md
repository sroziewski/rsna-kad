Subset: 481 studies (58 gold, seed-0 sample 400, audit 25); 5772 cells per table.

| table | model | rules | gold-58 AUC [CI] | parse% | verbatim% | rep/min | cell agr% | neg_unment | abs_vs_neg | subthr_neg | misattr | quote_miss | quote_fabr | audit-25 nu/sn/ma/qm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| awq_v1_base | Qwen/Qwen3-32B-AWQ | base | 0.850 [0.820,0.880] | 100.0 | 96.6 | 11.4 | 100.0 | 127 | 260 | 178 | 105 | 141 | 25 | 11/6/4/7 |
| awq_v2_base | Qwen/Qwen3-32B-AWQ | base | 0.838 [0.804,0.873] | 100.0 | 96.3 | 6.4 | 91.7 | 180 | 231 | 220 | 116 | 156 | 32 | 11/9/8/10 |
| rules | ? | ? | 0.816 [0.777,0.854] | 100.0 | 96.4 | nan | 82.0 | 0 | 0 | 0 | 56 | 147 | 0 | 0/0/4/5 |

Per-label gold AUC:

| table | acl | mcl | med_men | lat_men | med_oa | lat_oa | pf_oa | effusion | synovitis | baker | contusion | fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| awq_v1_base | 0.882 | 0.957 | 0.932 | 0.838 | 0.904 | 0.841 | 0.891 | 0.817 | 0.676 | 0.914 | 0.717 | 0.825 |
| awq_v2_base | 0.892 | 0.966 | 0.924 | 0.838 | 0.872 | 0.826 | 0.891 | 0.768 | 0.674 | 0.835 | 0.752 | 0.824 |
| rules | 0.868 | 0.912 | 0.838 | 0.812 | 0.791 | 0.736 | 0.828 | 0.816 | 0.658 | 0.910 | 0.749 | 0.871 |

Per-language gold AUC:

| table | en (n=28) | es (n=10) | tr (n=6) |
|---|---|---|---|
| awq_v1_base | 0.862 | 0.873 | 0.728 |
| awq_v2_base | 0.850 | 0.868 | 0.722 |
| rules | 0.821 | 0.821 | 0.702 |

awq_v2_base vs awq_v1_base: 91.7 % of cells identical; largest state changes (awq_v1_base->awq_v2_base): absent->negated 191, negated->absent 91, affirmed->negated 77, affirmed->absent 33, absent->affirmed 28, negated->affirmed 27, affirmed->hedged 18, hedged->affirmed 4

rules vs awq_v1_base: 82.0 % of cells identical; largest state changes (awq_v1_base->rules): absent->negated 260, negated->absent 249, negated->affirmed 205, affirmed->absent 115, affirmed->negated 110, hedged->affirmed 49, absent->affirmed 46, hedged->negated 4
