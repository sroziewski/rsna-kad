# RSNA Knee Abnormality Detection: project summary

Kaggle competition, 2026-09-17 → 2026-10-22. Predict 12 abnormalities from knee MRI (macro AUC). Team
**"Roziewski & Jiang & Tikhomirov"**: Szymon (Simon) Roziewski, Qile Jiang, Andrey Tikhomirov.

## Result

- **Final leaderboard place: 85 of 6,425 teams, public score 0.961, silver zone, 135 submissions** (place: AR
  L1661; zone and submissions: AR L1659). At the last check before the deadline (10-22 23:34 UTC) the team was
  rank 87 (AR L1658-1659). On 10-21 at 02:58 UTC the gold cut was 0.965 (rank 22) and the silver cut 0.958
  (rank 315), of 6,302 teams (AR L1594-1595).
- **Final picks** (both by Andrey, both 0.961):
  - **U8** 57519548 `rsna-knee-flat4-syn-r6-s1-stk6`: C3S + an out-of-fold stacker on 6 findings;
  - **C3S** 57455318 `rsna-knee-flat4-syn-r6-s1`: R6 + MT-hires + OL-plane + hx48 at 0.25 each, a Synovitis
    stacker, and test-time blur.
- Which board the final place 85 comes from (public or private) was not recorded.

## Contents

| Section | Covers |
|---|---|
| [01_data.md](01_data.md) | competition data, the 58 gold studies, folds, label versions, external datasets |
| [02_pipeline.md](02_pipeline.md) | our pipeline: report labelling with LLMs, targets, cache, KneeMIL model, Kaggle inference |
| [03_team.md](03_team.md) | what each teammate built, merge history, where each person's artefacts live |
| [04_timeline.md](04_timeline.md) | phases, LB progression, every recorded submission |
| [05_ensemble_finals.md](05_ensemble_finals.md) | the final ensemble member by member, and why everything tied at 0.961 |
| [06_lessons.md](06_lessons.md) | what worked, dead ends, decision rules, lessons, open gaps |
| [07_scripts_inventory.md](07_scripts_inventory.md) | every module, script and qa/ experiment directory with its result |

## The story in ten lines

1. Only 58 of 4,407 training studies had labels. Everything else had to be labelled from multilingual radiology
   reports.
2. Szymon labelled the reports with a rule voter plus Qwen3-32B (2 prompts), plus the pilkwang public-table voter,
   and calibrated the result on gold rows from the other folds.
3. The model is a ConvNeXt-V2-tiny 2.5D MIL over 5 MRI views (KneeMIL). Pseudo-labels (0.5 label + 0.5 teacher)
   were the biggest single lever. The solo pipeline peaked at **0.948**.
4. About 25 further solo ideas (DINOv3, DANN, nnPU, new heads, TTA, OAI pretraining...) all tied, under
   pre-declared gates.
5. Public models plus our leg reached **0.957**; plain R7 alone scored 0.956, so most of that came from the public models.
6. Qile (Opus-labelled ConvNeXt / ResNet models) and Andrey (native-DICOM CoAtNet MIL "hx48") merged on 10-14 and
   10-15.
7. Szymon's joint kernel ran each team's leg in its own process and rank-blended them: **0.959**.
8. Andrey's test-time blur sweep and out-of-fold stackers reached **0.961**.
9. Twelve submissions, B included, tied at 0.961, within the LB noise (SD ~0.003).
10. The finals were picked for upside (U8) and robustness (C3S).

## Glossary

| Term | Meaning |
|---|---|
| gold-58 | the 58 training studies with image-graded host labels: our only clean validation set (veto, never trained on) |
| OOF | out-of-fold predictions: each study predicted by the fold model that did not train on it |
| LB | public leaderboard (~30 % of the hidden test, 3 decimals); "full-precision rank" = order on the LB page, which ranks unrounded scores |
| UI order | the order Kaggle lists tied 0.961 submissions in; the only tiebreak visible to us |
| Cfull | an arm-C model trained on all 4,349 non-gold studies (`--fold full`) |
| g1 | Qile's "generation 1": trained on 0.5 labels + 0.5 previous-generation OOF (self-teaching) |
| silent cell | a (study, finding) the report never mentions; weak evidence for under-reported findings |
| EB / PAVA | empirical-Bayes calibration of label states to target values / pool-adjacent-violators to keep them ordered |
| K_infer | slices sampled per series at inference |
| legs V / P / A | parts of the joint kernel: V = our v3stack (R7 + Raptor + main_v5_C), P = Qile's notebook, A = Andrey's hx48 |
| R6 / R7 | public compact ConvNeXt-small models by heliosli (R7 has a fastMRI auxiliary) |
| Raptor | public CoAtNet models by nartaa (384 px; a 224 px variant only in the OAI stack), trained with OAI |
| S6 | another public model family by nartaa |
| pilkwang table | public LLM report-label table used as our 4th label voter |
| goodpjw reader | public 2.5D ConvNeXt notebook (tied as an extra leg) |
| MTR / MTH / OLP | Qile's MT-r50 / MT-hires / OL-plane models |
| hx48 (hx, n48, hx16) | Andrey's CoAtNet MIL pair; hx16 = his 5-fold variant whose OOF stood in for hx48 when fitting stacker5 |
| B, A, C3, C3S, C3SE, D, U1b-U9, flat4-7 | Andrey's / Qile's team blend names; contents in [04_timeline.md](04_timeline.md) and [05_ensemble_finals.md](05_ensemble_finals.md) |
| bpblur / bpsyn / bprefit | Szymon's variants of B: Qile leg blurred / + Synovitis stacker / stacker refit |

Internal note: server IP and `/raid/...` paths are for the team only. Strip them from any public write-up.

## Sources

The master log is `docs/architecture.md`; most numbers trace to it or to the `qa/<dir>/report.md`
files and the teammates' READMEs it cites. Data facts come from `docs/data_analysis.md`, literature and rules from
`docs/research.md`.
