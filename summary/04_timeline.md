# 4. Timeline and submissions

The full day-by-day record is `docs/architecture.md` §4 (L244-1661). This page is the condensed version. LB means
the public leaderboard (3 decimals); gold means macro AUC on gold-58.

## 4.1 Public LB best over time

| Date | Team best | What got there |
|---|---|---|
| 09-19 | 0.865 | MVP EffNetV2-S 256 |
| 09-19 | 0.931 | arm A ConvNeXt-S 384 |
| 09-23 | 0.933 | main_v3 = A + E + C |
| 09-27 | 0.943 | main_v4: pseudo-labels round 1 |
| 09-30 | 0.948 | main_v5_C: 6 full-data ConvNeXt-V2-T |
| 10-10 | 0.953 | public OAI stack 0.7 + main_v5_C 0.3 |
| 10-12 | 0.955 | v2stack |
| 10-13 | 0.957 | v3stack (R7 + Raptor 0.7, ours 0.3) |
| 10-15 | 0.959 | joint v2 = v3stack + Qile, 50/50 per finding |
| 10-19 | 0.960 | blur in team legs |
| 10-20 | **0.961** | Andrey's B (blur 0.288 mm + stacker5), then a plateau: 12 submissions (B included) tied at 0.961 |

Final: **place 85 of 6,425 teams, public score 0.961, silver zone, 135 submissions** (AR L1659, L1661). Earlier
snapshots: rank 82 on 10-20 / 10-21, and rank 87 at the last check before the deadline (10-22 23:34 UTC,
AR L1658-1659). On 10-21 at 02:58 UTC the gold cut was 0.965 (rank 22) and the silver cut 0.958 (rank 315), of
6,302 teams (AR L1594-1595).

## 4.2 Phases

1. **09-17 → 09-21: foundations.** Data fetch, EDA, cache, folds, rules + LLM labels, MVP, first submissions, arms
   A / E / C, ensemble analysis, decision rule (AR §4.2).
2. **09-22 → 09-30: label work and pseudo-labels.** Gemma voter, 8-epoch schedule, audit, pseudo-labels round 1
   (+0.0135 gold, +0.010 LB), full-data retrains, main_v5_C.
3. **10-01 → 10-09: plateau at 0.948.** Pipeline B (DINOv3), crop TTA, max-pool, nnPU, silentq, glmgrade, DANN,
   stale teacher, t70, multi-teacher: all ties or losses. Inference hardening after a timeout.
4. **10-10 → 10-13: public-model stacks.** Our C leg used as a diversity leg on top of public models, 0.953 → 0.957.
5. **10-14 → 10-18: merges and joint kernels.** Qile and Andrey join; joint / joint3 kernels; runtime limits
   (8.6-9 h) become the binding constraint; 0.959.
6. **10-19 → 10-22: blur, stackers, finals.** Andrey's blur sweep and stackers reach 0.961. Szymon ran blur / stacker
   probes (ties) and picked the finals.

## 4.3 All submissions recorded in `architecture.md`

### Szymon solo (09-19 → 10-13)

| Id | Date | Name | LB | Note |
|---|---|---|---|---|
| 56345216 | 09-19 | MVP effv2s@256 | 0.865 | first baseline |
| 56424502 | 09-19 | arm A convnext_s@384 | 0.931 | |
| 56433490 | 09-21 | eff_v1 (efficiency track) | 0.913 | efficiency track dropped later |
| 56433900 | 09-21 | main_v2 (A + eff) | 0.929 | |
| 56488491 | 09-23 | main_v3 (A + E + C, 15 members) | 0.933 | adopted |
| 56588465 | 09-26 | E_p pseudo probe | 0.930 | probe |
| 56588582 | 09-27 | main_v4 (pseudo round 1) | 0.943 | adopted |
| 56687124 | 09-30 | main_v5_bal | 0.945 | hedge |
| 56687125 | 09-30 | **main_v5_C** (6 Cfull) | **0.948** | adopted |
| 56710956 | 09-30 | main_v6 (12 Cfull) | 0.947 | tie |
| 56751245 | 10-01 | main_v5_C + crop TTA | 0.947 | tie |
| 56803029 | 10-03 | main_v5_C 12 seeds | 0.947 | tie |
| 56804636 | 10-03 | + max-pool + nnPU | 0.948 | tie |
| 56808088 | 10-04 | + 0.2 pipeline-B on 4 labels | 0.947 | tie |
| 56837972 | 10-05 | + silentq synovitis | error | notebook timeout → watchdog added |
| 56854249 | 10-05 | same, hardened | 0.948 | final candidate then |
| 56857914 | 10-05 | main_v5_C hardened | 0.948 | OAI-free hedge until 10-14 |
| 56874956 | 10-06 | t70 | 0.944 | |
| 57022888 | 10-10 | v5c + goodpjw reader | 0.948 | tie |
| 57037844 | 10-10 | v5c-oaistack | 0.953 | adopted |
| 57070607 | 10-11 | stk-oaistack | 0.952 | |
| 57084313 | 10-12 | v2stack | 0.955 | adopted |
| 57134537 | 10-13 | OAI-trained C in v2stack | 0.955 | tie, OAI closed |
| 57150688 | 10-13 | plain R7 (heliosli) | 0.956 | baseline |
| 57150980 | 10-13 | **v3stack** | **0.957** | adopted |
| 57193974 | 10-14 | v4pf (per-finding C weights) | not recorded | |

### Team (10-15 → 10-22)

| Id | Date | Name | Author | LB | Note |
|---|---|---|---|---|---|
| 57127512 | ≤10-15 | mtr1-mth4-olp5 | Qile | 0.957 | partner leg P |
| 57225242 | 10-15 | **joint v2** (V .5 + P .5) | Szymon | **0.959** | main pick until 10-20 |
| 57236870 / 57236838 | 10-15 | hx / mixed v3 | Andrey | 0.945 / 0.948 | |
| 57245439 | 10-15 | joint3 v1 (.4/.4/.2) | Szymon | 0.959 | leg A had silently timed out |
| 57261640 | 10-16 | joint3 require-all-legs | Szymon | failed | a leg was dropped |
| 57277456 | 10-16 | joint3diag | Szymon | 0.840 | diagnostic: code 3 = leg A timeout |
| 57283695 | 10-17 | flat5 v6 | Qile | 0.959 | |
| 57298855 | 10-17 | blend5 v1 | Qile | not recorded | |
| 57325519 | 10-17 | P + A (no V) | Szymon | 0.958 | |
| 57326594 | 10-17 | joint-extc (MRNet/KneeMRI C) | Szymon | 0.959 | tie |
| 57327062 | 10-17 | joint3q3 (all 3 legs ran) | Szymon | 0.959 | |
| 57332429 | 10-18 | flat7 | team (posted by Qile) | 0.959 | |
| 57361733 | 10-18 | ConvNeap (ConvNeXt-B) | Andrey | 0.943 | |
| 57368898 | 10-18 | joint3q3 equal thirds | Szymon | 0.958 | |
| 57370104 | 10-18 | flat6 | Qile | 0.959 | |
| 57372562 | 10-18 | nested958 | team ("peach" in AR L1409) | 0.958 | |
| 57403485 | 10-19 | joint3s6r | Szymon | error | |
| 57405419 | 10-19 | joint3blur (P / A 0.65 px) | Szymon | 0.960 | |
| 57409437 | 10-19 | joint3blurall | Szymon | error | time budget |
| 57411614 | 10-20 | A flat7-blur | Andrey | 0.960 | final pick 10-20 → 10-21 |
| 57411322 | 10-20 | **B j3-stack-blur** | Andrey | **0.961** | first 0.961; final pick until 10-22 |
| 57432197 | 10-20 | joint3blurall2 (8.8 h budget) | Szymon | 0.960 | |
| 57438367 | 10-20 | C3 flat4-syn-r6 | Andrey | 0.961 | |
| 57439962 | 10-20 | bpblur | Szymon | 0.961 | tie |
| 57452940 | 10-20 | D flat5 | Andrey | 0.961 | |
| 57455318 | 10-20 | **C3S** flat4-syn-r6-s1 | Andrey | 0.961 | **FINAL** |
| 57463786 | 10-21 | bpsyn | Szymon | 0.961 | tie |
| 57485680 | 10-21 | U2 (anti-aliased warp) | Andrey | 0.961 | |
| 57487018 | 10-21 | U1b | Andrey | 0.961 | |
| 57487526 | 10-21 | U5 | Andrey | 0.961 | |
| 57491297 | 10-21 | C3SE (Eff/Syn expert) | Andrey | 0.961 | below C3S in UI order, dropped |
| 57518756 | 10-22 | U4b | Andrey | no score | |
| 57519548 | 10-22 | **U8** flat4-syn-r6-s1-stk6 | Andrey | 0.961 | **FINAL** |
| 57521316 | 10-22 | U7 | Andrey | no score | |
| 57533566 | 10-22 | U9 flat4-syn-r6-stk6-50 | Andrey | 0.961 | tie, not swapped in (AR L1658) |

Final picks over time: 56854249 + 56857914 (10-05) → v3stack / joint (10-13 → 10-15) → B + A (10-20) → B + C3S (10-21)
→ **U8 + C3S** (10-22 ~20:20 UTC).
