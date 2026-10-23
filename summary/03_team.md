# 3. Team and contributions

Team **"Roziewski & Jiang & Tikhomirov"**. Szymon (Simon) Roziewski (`sroziews`, owner of this repo) worked solo
until 10-14. Qile Jiang (`qilejiang76`) merged on **2026-10-14** and Andrey Tikhomirov (`andreytikhomirov`) on
**2026-10-15** (AR L1163, L1228). After the merge the team had one shared budget of 5 submissions a day. Team chat
was on Discord.

## 3.1 Szymon Roziewski

**Own pipeline (09-17 → 10-10)**, details in [02_pipeline.md](02_pipeline.md):
- Data fetcher for the 570 GB dataset, the preprocessing cache, site-grouped folds, and the gold-58 protocol.
- Report → label extraction: rules → Qwen3-32B-AWQ (2 prompts) → 4-voter merge (+ pilkwang public table; Gemma-3-27B was the clean-provenance alternative),
  then fold-local calibration.
- Model arms A (ConvNeXt-S 384), E (EffNetV2-S 256), C (ConvNeXt-V2-T FCMAE 384, 5 views). Pseudo-labels round 1
  (the biggest lever, +0.010 LB). Full-data retrains, ending in **main_v5_C** (6 Cfull seeds, LB 0.948, gold-58
  0.9254 for the 6-seed mean; 0.9253 from the end-to-end kernel run).
- Kaggle inference kernel with a watchdog against hangs, plus the export, parity and liveness tooling.
- About 25 rejected experiments, each with a pre-declared gate (see [06_lessons.md](06_lessons.md)).

**Public-model stacks (10-10 → 10-13):** main_v5_C rank-blended with public notebooks. OAI stack 0.953 → v2stack
0.955 → **v3stack** (heliosli R7 + nartaa Raptor 0.7, ours 0.3) **0.957**.

**Team infrastructure (10-14 → 10-22):**
- `qa/joint/build_joint.py` + `joint_template.py`: one kernel that runs several teams' legs as separate processes
  and rank-blends them per finding. It has failure fallbacks, `--require-all-legs`, `--diag-encode` (encodes which
  leg failed into the score) and `--total-h` (time budget). This built joint v2 (**0.959**, team best on 10-15),
  joint3 and joint3q3.
- Ran Andrey's OOF kernels on the GPU server (blurred MT-hires and R6/R7 OOF for 4,349 studies).
- Probes: ext-C, r6base, s6r, Opus-label C, leg blur (joint3blur 0.960, joint3blurall error, joint3blurall2 0.960), bpblur / bpsyn (0.961 ties), bprefit.

## 3.2 Qile Jiang

**Before the merge:** solo 0.957 (AR L1163-1166).

**Labels:** "Opus Blend" soft labels (`labels_opus_blend.csv`, in `Qile's labels and predictions.zip`). They come from
Claude Opus 5.5 run 3 times and averaged, with a product-of-experts confidence per cell, blended with soft
pseudo-labels; the blend weights were tuned on gold-58. Scored as labels on gold-58 they reach **0.9364**, against
0.8728 for our calibrated v1.1 *targets* (AR L1211; the raw v1.1 voter merge in [01_data.md](01_data.md) is 0.866). Retraining our C leg on them gave a tie (`qa/opus`).

**Models** (0.957 submission 57127512, kernel `qilejiang76/rsna-knee-ours-mtr1-mth4-olp5`). The blend is a per-finding
rank blend of 0.1 / 0.4 / 0.5, with 5 folds per member:

| Member | Model | Labels | Gold-58 |
|---|---|---|---|
| **MTR** = MT-r50 | ResNet-50 (RadImageNet), 224 px | mean of label tables + one self-teaching round (g1) | 0.9195 |
| **MTH** = MT-hires | ConvNeXt-T, 384 px, 120 mm crop | mean of tables, g1 | 0.9304 |
| **OLP** = OL-plane | 2 ConvNeXt-T 384 plane specialists (sag+ax, cor) merged per finding | "Opus lite", g1 | 0.9372 |
| Blend | | | 0.9399 |

g1 = trained on 0.5 labels + 0.5 previous-generation OOF. One leg uses OAI.

**After the merge:**
- Full-data (FULL) retrains: OL-plane-FULL and MT2T-hires-FULL in AR L1336-1338; OL-hires-FULL and MT2T-plane-FULL per team chat.
- A ~1.9x faster notebook (v3, decoding overlapped with the GPU, same predictions).
- Team blends flat5 (0.959), flat6 (0.959) and blend5; he also posted the team blend flat7 (0.959).
- Datasets: `qilejiang76/rsna-knee-lineage-g1-mth-mtr-olp-20261013`, `qilejiang76/rsna-knee-single-models-20261007`.

## 3.3 Andrey Tikhomirov

**Before the merge:** solo 0.955, 34 submissions.

**hx48** (kernel `andreytikhomirov/rsna-knee-native-hx48`, LB 0.948, gold 0.936 as scored; 0.937 for the last-epoch file; 0.9365 as our joint leg; qa/andrey/README.md, AR L1241-1242, L1288):
- His own native DICOM pipeline. YOLOX-nano knee landmarks run on the CPU. Series go into typed slots, with 160 mm
  wide and 96 mm knee-centred crops at 320 px, and inference runs on every slice (`qa/andrey/README.md`).
- Two **CoAtNet-RMLP-2 MIL** models with SWA checkpoints:
  - **hx**: 12 positions, metadata embeddings, canonical side; trained on competition data plus MRNet and KneeMRI
    pseudo-labels. Public 0.945.
  - **n48**: 48 positions, competition data only.
- Per-finding rule: Fracture / Lat Men / Lat OA come from n48, Synovitis / Effusion from hx, and the rest are the
  mean of ranks.
- Hidden-test runtime 106 min; v6 brought it to ~50 min with the same checkpoints.
- Datasets: `andreytikhomirov/rsna-knee-native-hx48-weights`, `andreytikhomirov/rsna-knee-landmark-detector`.

**Techniques he brought:**
- **Test-time blur sweep** (8 levels, 0-0.504 mm, 11 models). At **0.288 mm**, R7 gains +0.0101 on gold (P 99.8 %),
  MT-hires +0.0067 and S6 +0.0058; our v5-C −0.0005. Blur is applied at the DICOM level through a pydicom hook
  (`sitecustomize`), and that held up on the hidden test.
- **Out-of-fold stackers:**
  - **stacker5** (in B): 60 per-member rank features; per finding, LR and pairwise RankLR (C = 0.1) are combined as
    0.5 / 0.5 ranks, then block = 0.8 anchor + 0.2 stack. Fitted on 4,349 OOF studies with curated labels
    (`.tmp/andrey_stacker5_zip/stacker5_for_simon/fit_stacker5.py`).
  - **S36** Synovitis stacker.
  - **S48** stacker on 6 findings (in U8).
  - stacker5's hx48 features came from his hx16 5-fold OOF as a stand-in (`fit_stacker5.py` L7).
  - S36 / S48 internals come from Andrey's kernel descriptions on Kaggle; they are not in this repo.
- **Seed replicate** in the hx slot (C3S). **Expert model** for Effusion/Synovitis (C3SE). **Anti-aliased crop warp**
  (U2).
- **Curated external labels** `hybrid_curated_mrnet_v4_kneemri10_labels.csv` (6,344 rows).
- **Validation insight:** gold-58 has no normal knees, so it under-rates models that help on normal anatomy. Models
  trained with extra knees score ~0.025 higher on LB than on gold (AR L1246-1249). Afterwards the judges for
  extra-knee legs treated gold as information only, with a floor of −0.004 (AR L1316-1317).

**Submissions:** B, C3, C3S, D, U1b, U2, U5, C3SE, U8 and U9, all 0.961. The only other 0.961s were Szymon's
bpblur and bpsyn. Both final picks are Andrey's (see [05_ensemble_finals.md](05_ensemble_finals.md)).

## 3.4 Merge history

Other teams are anonymised here. Teams 3-5 are recorded in AR L1195-1201 and teams 7 and 9 in AR L1250; team 7's gold numbers are in its proposal
file in the repo root. Teams 1, 2, 6 and 8 come from session notes only.

| Date | Candidate | Offer | Outcome |
|---|---|---|---|
| 10-12/13 | team 1 | 0.954 incl. public models | declined 10-14 in favour of Qile |
| 10-12 | team 2 | 0.954 | declined |
| 10-14 | **Qile Jiang** | 0.957, own models, Opus labels | **accepted** |
| 10-15 | team 3 | 0.956 | gold-58 predictions requested first; no merge |
| 10-15 | team 4 | 0.957 | gold-58 files sent; adding them lowered our gold, −0.0036 at weight 0.25 (AR L1198-1199); declined |
| 10-15 | team 5 | 0.942 | declined |
| 10-15 | **Andrey Tikhomirov** | 0.955, own hx48 0.948 | **accepted** (+0.0018 gold, P 0.83) |
| 10-15 | team 6 | 0.963 | in contact; no merge before the deadline |
| 10-15 | team 7 | 0.957; sent gold-58 OOF of its own models (0.898-0.908) | declined: weaker than our members |
| 10-15 | team 8, team 9 | 0.957 / 0.956 | no merge |

Merge rule (AR L1197): merge only at > +0.003 on gold, when the runtime fits, and with the existing members agreeing.
Andrey's merge was an exception to the gold bar: +0.0018 is below +0.003, but he was the most independent candidate
(Spearman 0.916 with the joint) and merged anyway (AR L1226-1228).

## 3.5 Where each person's artefacts live

| Who | Local | Kaggle |
|---|---|---|
| Szymon | the whole repo; `qa/joint`, `qa/blur`, `qa/bblur`, `qa/bprefit`; `share/` | kernels `sroziews/rsna-kad-*` (joint, joint3, joint3q3, joint3blur*, bpblur, bpsyn, main-v5-c) |
| Qile | `Qile's labels and predictions.zip`, `qa/qile/` | notebook `qilejiang76/rsna-knee-ours-mtr1-mth4-olp5`; datasets above; kernel `rsna-kad-flat7` |
| Andrey | `hx48_ensemble.zip`, `qa/andrey/`, `.tmp/andrey_stacker5_zip/`, `.tmp/for_andrey/` (OOF we made for him), `files/` (curated labels) | kernels `rsna-knee-native-hx48`, `rsna-knee-j3-stack-blur`, `rsna-knee-flat7-blur`, `rsna-knee-flat4-syn-r6-s1` (+ `-stk6`, `-es2`), `rsna-knee-flat4-syn-r6-stk6-50`, `rsna-knee-mth-blur-oof`, `rsna-knee-r6-r7-blur-oof`. The S48 stacker JSON (`research/d_stacker_s48_6.json`) is in Andrey's own repo |
