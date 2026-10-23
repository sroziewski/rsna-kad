# 5. Final ensemble

## 5.1 The two final picks

| Pick | Submission | Kernel (Andrey) | LB | Role |
|---|---|---|---|---|
| U8 | 57519548 | `rsna-knee-flat4-syn-r6-s1-stk6` v1 | 0.961 | upside: per Andrey, +10 LB positions over our other 0.961s at full precision |
| C3S | 57455318 | `rsna-knee-flat4-syn-r6-s1` v1 | 0.961 | hedge: flat 0.25 weights, few tuned parts, no gold search |

Both belong to Andrey's C3 lineage. No model Szymon trained is in either final: main_v5_C left the picks when U8
replaced B on 10-22.

## 5.2 What the members are

| Name | What it is | Who trained it | In finals? |
|---|---|---|---|
| **R6** | heliosli public compact ConvNeXt-small (bundle fe055406), flip TTA off | public | yes (C3 / C3S / U8) |
| R7 | heliosli public compact ConvNeXt-small (bundle 8c7b666e, 4 folds, crop 352, fastMRI aux) | public | no (in B / D / v3stack) |
| Raptor | nartaa CoAtNet SWA, trained with OAI; 384 px in v2stack / v3stack, 384 + 224 in the OAI stack | public | no |
| **MTH** | Qile's MT-hires ConvNeXt-T 384, 5 folds, g1 | Qile | yes |
| MTR | Qile's MT-r50 ResNet-50 224 | Qile | no (in B) |
| **OLP** | Qile's OL-plane, 2 ConvNeXt-T plane specialists | Qile | yes |
| **hx48** | Andrey's CoAtNet-RMLP-2 MIL pair hx + n48, YOLOX localisation, per-finding rule | Andrey | yes |
| v5c | our main_v5_C, 6 ConvNeXt-V2-T full-data seeds | Szymon | no (in B, v3stack, joint) |
| stacker5 | out-of-fold LR + RankLR over MTR / MTH / OLP / v5c / hx48 ranks | Andrey | no (in B) |
| **S36** | out-of-fold Synovitis stacker on MTH / OLP / hx48 ranks | Andrey | yes |
| **S48** | out-of-fold stacker on R6 / MTH / OLP / hx48 ranks, refit for 6 findings | Andrey | U8 only |

## 5.3 How they are combined

Every blend is a **per-finding rank average**. AUC depends only on ranks, so ranks make members on different scales
comparable (rank blends: AR L1058, L1170, L1236).

```
C3   = mean-rank( R6 [blur 0.288 mm, no flip],
                  MTH [blur 0.8 px],
                  OLP,
                  hx48 )                      0.25 each
       Synovitis = 0.75 rank(C3 blend) + 0.25 rank(S36 stacker)
C3S  = C3, with hx48's hx slot = rank of the mean rank of (hx, hx seed-1 replicate)
U8   = C3S, then for PF OA, Lateral OA, Baker's, ACL, Medial OA, Fracture:
       0.75 rank(C3S) + 0.25 rank(S48 stacker)          (research/d_stacker_s48_6.json)
```

All weights were fixed before submitting. The expected hidden-test runtime for C3 is ~5.2-5.6 h of the 9 h limit (AR L1586, L1604);
it was not measured separately for C3S / U8.

Source note: the repo records U8 = "C3S + S48 OOF stacker on 6 findings at share .25" and C3S = "C3 + Synovitis
stacker" (AR L1590, L1654). The finer detail above comes from Andrey's kernel descriptions on Kaggle, as pasted
into the team chat. That covers the six findings, the S48 inputs, the seed-1 construction, and the 0.8 px / 0.36 mm
grid. None of it is verified against code in this repo.

**Blur.** Gaussian blur at test time is applied only to the members where Andrey's gold-58 sweep showed a gain:
- 0.288 mm at the DICOM level on R6 / R7, through a pydicom `sitecustomize` hook;
- 0.8 px of the 0.36 mm grid on Qile's hires views.

The likely mechanism is aliasing in the resampled crops (U2 added an anti-aliased crop warp to hx48 on top of D flat5 and tied).

## 5.4 The predecessor B (final pick until 10-22)

`B j3-stack-blur` (57411322, 0.961, gold 0.9509) regrouped the joint3q3 members:
`0.28 rank(R7) + 0.72 rank(0.8 anchor + 0.2 stacker5)`, with blur on R7 and MTH. It was the only final candidate
containing our v5c leg and R7, and so the most different from the C3 family. The team traded that diversity for
U8's higher full-precision rank.

## 5.5 Why everything tied at 0.961

- The public LB covers ~30 % of 1,322 studies, and the SD of one submission is ~0.003 (forum and the leoprovorov
  write-up; AR L1347, L1597).
- Gold-58 transfers to LB at a slope of 0.24, so +0.001 on gold is about +0.0003 on LB (AR L1536-1539).
- The adoption bar was therefore **≥ 0.962**, and nothing reached it. Ties could only be ordered by the UI order
  and Andrey's estimate from leaderboard positions.
- Kaggle staff warned that public blends tend to drop on the private LB. That is the argument for keeping the
  less-tuned C3S as the hedge.
