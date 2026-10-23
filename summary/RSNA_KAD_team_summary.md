<a id="overview"></a>

# RSNA Knee Abnormality Detection: project summary

**Sections:** [Overview](#overview) · [Data](#data) · [Pipeline](#pipeline) · [Team](#team) · [Timeline](#timeline) · [Finals](#finals) · [Lessons](#lessons) · [Inventory](#inventory)


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
| [Data](#data) | competition data, the 58 gold studies, folds, label versions, external datasets |
| [Pipeline](#pipeline) | our pipeline: report labelling with LLMs, targets, cache, KneeMIL model, Kaggle inference |
| [Team](#team) | what each teammate built, merge history, where each person's artefacts live |
| [Timeline](#timeline) | phases, LB progression, every recorded submission |
| [Finals](#finals) | the final ensemble member by member, and why everything tied at 0.961 |
| [Lessons](#lessons) | what worked, dead ends, decision rules, lessons, open gaps |
| [Inventory](#inventory) | every module, script and qa/ experiment directory with its result |

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
| B, A, C3, C3S, C3SE, D, U1b-U9, flat4-7 | Andrey's / Qile's team blend names; contents in [Timeline](#timeline) and [Finals](#finals) |
| bpblur / bpsyn / bprefit | Szymon's variants of B: Qile leg blurred / + Synovitis stacker / stacker refit |

Internal note: server IP and `/raid/...` paths are for the team only. Strip them from any public write-up.

## Sources

The master log is `docs/architecture.md`; most numbers trace to it or to the `qa/<dir>/report.md`
files and the teammates' READMEs it cites. Data facts come from `docs/data_analysis.md`, literature and rules from
`docs/research.md`.

<a id="data"></a>

## 1. Data

Sources: `docs/data_analysis.md` (DA), `docs/research.md` (RS), `docs/architecture.md` (AR, cited as `AR L<n>`),
`analysis/out/*`. Raw numbers come from `analysis/eda.py`; the fact dump is `analysis/out/report_facts.txt`.

### 1.1 Competition data

| Item | Value |
|---|---|
| Files | 819,635 DICOM (819,078 train + 557 test), ~570 GB, 100 % Explicit VR Little Endian, uncompressed |
| Studies | 4,407 train (one patient per study, no patient leakage); test.csv = 3 placeholder studies |
| Hidden test | ~1,322 studies; public LB = ~30 %, private = ~70 % (AR L1597) |
| Series | 24,371 (median 5.5 per study, range 3-14) |
| Series types | `Fluid_Sensitive == Fat_Suppression` in every row, so 6 types: Sag FS0 / Sag FS1 / Cor FS0 / Cor FS1 / Ax FS0 / Ax FS1 |
| Geometry | median 30 slices (11-320), 3.4 % 3D series, pixel spacing median 0.312 mm, FOV median 160 mm, slice 3.0 mm, 221 matrix shapes |
| Vendors | 12 manufacturer strings, 46 models; BitsStored 12 (Siemens/Philips/Fujifilm/Hitachi) vs 16 signed (GE/Canon/Toshiba); raw intensity differs ~30x between vendors |
| Sites | all site tags stripped; proxy `site_key` = manufacturer + model + field + software + language, 123 groups |
| Laterality | DICOM tag usable in only 49 % of studies; 25 studies contradict L/R |
| Reports | one free-text radiology report per study, 9 languages: en 39 %, es, tr, hr, el, de, bg, nl, fr; language is nearly a function of the scanner |

**The 12 findings** (train.csv order): ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
Effusion, Synovitis, Baker's cyst, Contusion, Fracture. **Metric:** macro ROC AUC over the 12.

**The core difficulty:** only **58 of 4,407** training studies (1.3 %) carry labels. The host's radiologists graded
those 58 and the hidden test set from the images. The other 4,349 studies have to be labelled from their reports,
and reports under-report: synovitis is affirmed in 12.5 % of reports but positive in 46.6 % of gold studies, and
fracture in 5.1 % against 31 % (AR L285).

### 1.2 Gold-58 (the only image-graded labels we had)

- Ids in `data/labels/gold_ids.txt` (= `is_gold` in `folds_v1.csv`). Gold-58 was **never trained on**, and it was
  never used as few-shot examples. The leakage audits in `qa/stage3` and `qa/stage5` passed.
- Used as a **veto**, not a score: paired noise between ensembles ±0.007-0.009, level CI ±0.027 (AR §4.5). Over 54
  submissions the LB-vs-gold slope was only 0.24 above gold 0.94 (AR L1536-1539).
- Prevalence (positives/58): ACL 24, MCL 9, Med Men 26, Lat Men 23, Med OA 15, Lat OA 11, PF OA 21, Effusion 35,
  Synovitis 27, Baker's 12, Contusion 19, Fracture 18. It is enriched compared with the population and holds **no
  normal knees** (Andrey, AR L1246).
- Gold-prediction files: `latest_gold.csv` (Andrey's hx48 predictions on gold-58, *not* labels),
  `data/gold58_hx48_v6.csv`, `share/szymon_gold58_predictions/` (main_v5_C, R7 base, v3stack), and
  `merge_share_2026-10-15_gold58.zip` (from a merge candidate).

### 1.3 Folds

`data/folds/folds_v1.csv`: 5 folds of 879-883 studies, grouped by `site_key`, with 11-12 gold studies per fold. Built
by a seeded greedy balancer (`cv.make_folds(method="balanced")`), because sklearn `StratifiedGroupKFold` failed the size
and gold criteria for all 200 seeds tried. Full-data runs train on all 4,349 non-gold studies and are scored on
gold only. Qile's g1 models use folds_v1 too (AR L1638). The heliosli R6/R7 OOF uses hash folds (sha1(uid) % 4).

### 1.4 Label versions

Each row shows the AUC of the label table **itself** on gold-58.

| Version | What | Gold-58 |
|---|---|---|
| `rules_v0` | multilingual rule extractor, 4 states (affirmed / hedged / negated / absent) | 0.816 |
| `llm_v1` | Qwen3-32B-AWQ, two prompts merged, 3 few-shots per language (`data/fewshot/`) | 0.852 |
| targets v0 / v1 | state → soft target + per-cell weight; fold-local empirical-Bayes calibration on training-fold gold + PAVA | - |
| `v1_1` | + pilkwang public LLM table as 4th voter (merge +0.0149, P 0.998) | 0.866 |
| `v1_1_gemma3` | own Gemma-3-27B voter instead of the public table (clean provenance, +0.0118) | - |
| Claude Sonnet subset | 481 studies; marginal +0.0043 over 3 voters, so the full pass was not run | - |
| **`v3_pseudo`** | **0.5 × v1_1 + 0.5 × main_v3 OOF teacher**, silent-cell weight 0.1 → 0.5 | **0.915** |
| `v3_pseudo_full` | same recipe for full-data training; the main_v5_C targets | - |
| variants | `_silentq`, `_glmgrade`, `_t70`, `_mt`, `_stk`, `_oai`, `_agree`, `_opus`, `_ext` | all ties/rejects |
| Opus Blend (Qile) | Claude Opus 5.5, 3 runs averaged, soft [0.05, 0.95] + per-cell confidence; blend weights tuned on gold | 0.9364 |
| hybrid_curated v4 (Andrey) | 6,344 rows = 4,349 competition + 1,250 MRNet + 745 KneeMRI, plus cell weights | - |

### 1.5 External data

| Dataset | Used by | Outcome |
|---|---|---|
| OAI (2,439 MOAKS knees, 75 GB, NDA package) | Szymon (`qa/oai`) | own OAI-trained C leg: LB tie 0.955, closed. OAI is also inside the public nartaa Raptor weights and one of Qile's legs |
| MRNet (Stanford AIMI, 1,250 exams) | Andrey (hx model), Szymon (`qa/ext`) | in hx48; ext-C LB 0.959 tie |
| KneeMRI (Zenodo, CC BY-NC-ND 4.0, 745 series) | Andrey, Szymon | as MRNet |
| fastMRI | heliosli R7 public model (auxiliary) | inside R7 |

Host rulings: external data is OK when it needs only registration and a licence acknowledgement (post 743416,
10-14). Hosted LLMs are allowed for labelling (post 733965). External data has to be disclosed in the write-up.

#### What the competition write-up must disclose

The external data and weights inside the final picks:
- **public weights:** heliosli R6 (and R7, which used fastMRI as an auxiliary, in B); nartaa Raptor (trained with
  OAI) in earlier picks;
- **Qile's models:** one leg uses OAI;
- **Andrey's hx model:** MRNet (Stanford AIMI) and KneeMRI (Zenodo, CC BY-NC-ND 4.0);
- **hosted LLMs used for labels:** Claude Opus (Qile's labels), the pilkwang public LLM table, Qwen3-32B (ours,
  self-hosted). Gemma-3-27B was tried but is not in the final labels.

### 1.6 Where the data lives

- Server copy: `/raid/sroziewski/data/rsna_kad/` (CRC-verified), fetched by `scripts/kaggle_fetch.py`
  (range-resumable, extracts while downloading; see `README.md`).
- Preprocessing cache v1 (server): 24,386 series, 98.2 GiB, uint8 (N, 384, 384), 130 mm crop, built in 265 s.
- Repo: `data/labels/`, `data/targets/v0|v1/`, `data/folds/`, `data/fewshot/`, `data/oof_train_armC_5fold.csv`;
  EDA tables in `analysis/out/` (~40 CSV/JSON plus `headers.parquet`, one row per DICOM file).
- `train_files.zip` in the repo root is a **0-byte file**, not an archive.

---

<a id="pipeline"></a>

## 2. Our pipeline (Szymon): reports → labels → models → Kaggle

The code is in `src/rsna_kad/` and `scripts/`. `pipelineA.py` is a single-file reproduction of the shipped
main_v5_C recipe. The design record is `docs/architecture.md` §1-3.

Note: the local mirror is partly behind the server copy (`/raid/sroziewski/sandbox/rsna_kad`). For example,
`infer.py` imports `FAT_COL`, which the local `series.py` does not define yet. Where they differ, the server copy is
the reference.

### 2.1 Flow

```
kaggle_fetch.py ─► DICOM on /raid (570 GB)
   │
   ├─► reports ─► rules voter ─┐
   │              Qwen3-32B V1 ├─► merge (+ public-table voter)   ─► targets (calibrated, weighted)
   │              Qwen3-32B V2 ┘                                              │
   │                                                    pseudo-labels: 0.5 label + 0.5 teacher OOF
   │                                                                          │
   └─► build_cache.py ─► uint8 volumes per series ─► KneeMIL training (5 folds / full data) ─► OOF, gold-58
                                                                          │
                                export_kaggle.py ─► weights dataset + generated notebook ─► Kaggle kernel ─► submit
```

### 2.2 Label extraction (`src/rsna_kad/labels/`)

- **Schema** (`schema.py`): for each of the 12 findings, a state (affirmed / hedged / negated / absent), a severity
  and an evidence quote of at most 120 characters; plus `knee_side`, `postoperative` and `normal_template`. Used as
  the vLLM structured-output JSON schema.
- **Rule voter** (`rules.py`, 864 lines): 9-language vocabularies, a negation window of 5 content tokens, a severity
  window of 6. Gold-58 0.816.
- **LLM voters** (`llm.py`, `prompts.py`): `Qwen/Qwen3-32B-AWQ` on vLLM, temperature 0, thinking off, in its own venv
  `.venv-llm` on GPU 0.
  - V1 = definitions first; V2 = report first plus a checklist.
  - Prompts carry the host's thresholds verbatim. Few-shots: 3 per language, hand-edited, with `assert_no_gold`.
  - Runs are resumable through JSONL with retry on parse failure.
  - Gold-58: V1 0.850, V2 0.838, merged 0.852.
- **Merge** (`merge.py`): disagreement is kept, not majority-voted. An evidence check fuzzy-matches the quote back
  into the report. Extra voters come in through `add_voter`: the pilkwang public table via `public.py`, or our own
  Gemma-3-27B. Mistral-24B was tried as well.
- **Rejected:** the "strict" prompt (it turned negations into "absent"; worse on gold, p = 0.03), bf16 / FP8
  precision (no gain over AWQ), and a Claude Sonnet voter (+0.0043 marginal).

### 2.3 Targets (`labels/targets.py`, `labels/pseudo.py`)

- Base values per state: affirmed 0.92, hedged 0.5, negated 0.04, absent 0.06; absent on an under-reported finding
  0.15; mild values per finding.
- **Calibration:** per fold, an empirical-Bayes posterior (prior strength 8) computed on the gold rows of the *other*
  folds, then weighted PAVA (affirmed ≥ hedged ≥ absent ≥ negated).
- **Weights:**
  - by state: affirmed / negated 1.0, hedged 0.7, absent 0.5, absent on an under-reported finding 0.1;
  - for problem rows: parse failure 0.2, postoperative 0.2, hallucinated quote capped at 0.3;
  - multiplied by a voter-disagreement factor.
- **Pseudo-labels v3 (shipped):** every non-gold cell = 0.5 × label + 0.5 × teacher. The teacher is the mean of the
  OOF sigmoids of main_v3. The weight on silent cells rises from 0.1 to 0.5. `v3_pseudo_full` calibrates on all 58
  gold rows for full-data training. Round 2 (students as teacher) made things worse through confirmation bias.

### 2.4 Folds (`cv.py`)

5 folds, grouped by `site_key` and stratified by language, built with the `balanced` method (best of 200 seeded
greedy assignments, seed 2026). The md5 of `data/folds/folds_v1.csv` is pinned in `tests/test_cv.py`. Gold rows get
a fold but are excluded from every training set (asserted in `train.py`).

### 2.5 Preprocessing cache (`dicom.py`, `series.py`, `preprocess.py`, `cache.py`)

- Per series: sort by InstanceNumber → apply rescale → fill unreadable slices from their neighbours → 130 mm centre
  crop → average 3D slabs to ~3 mm → take ≤ 40 slices → resize to 384 px → p1/p99 window → uint8 `.npy`.
- **Slots** come from (plane, fluid-sensitive): sag_fs, cor_fs, ax_fs, sag_t1, cor_t1, ax_t1, with an FS → T1
  fallback.
- **Laterality** (`laterality.py`) is estimated from the DICOM tag, the series description and geometry, and stored,
  but the models never flip or canonicalise the image (measured gain < 0.001).
- 24,386 series, 98.2 GiB, built in 265 s on 24 workers.

### 2.6 Model (`model.py`, `train.py`)

- **KneeMIL:**
  - a timm 2D backbone runs on 2.5D slice triplets;
  - each slot gets a gated-attention MIL pooling, with a learned vector for missing slots;
  - fusion is concatenation + presence bits → MLP → 12 logits;
  - per-slot auxiliary heads carry loss weight 0.3.
- **Training:** AdamW (backbone 1e-4, heads 1e-3), cosine schedule with warmup, EMA 0.999, bf16, manifold mixup
  p 0.3, weighted soft BCE, 8 epochs, last-epoch selection (best-epoch selection added ~+0.008 optimism).
  `--fold full` trains on all 4,349 non-gold studies.
- **Augmentation:** random crop scale, ±8° rotation, small contrast and brightness jitter, sibling-series swap
  p 0.3, slot dropout p 0.1. MRI intensity augmentation exists but is off (no gain).

| Arm | Backbone | Input | Outcome |
|---|---|---|---|
| MVP / E | tf_efficientnetv2_s | 256, 3 slots | LB 0.865; efficiency entry 0.913 / 0.930 |
| A | convnext_small.fb_in22k_ft_in1k_384 | 384, 4 slots | LB 0.931 |
| B | coatnet_rmlp_2_rw_384 | 384 | aborted: 3.7x slower, underfit |
| **C** | **convnextv2_tiny.fcmae_ft_in22k_in1k_384** | 384, 5 views (+ sag_fs 90 mm crop) | **the shipped family** |
| D | ConvNeXt-V2-T, cor_fs 100 mm 5th view | 384 | adds nothing |
| N | head experiments (per-label attention, residual, transformer) | 384 | closed: tie |
| Pipeline B | DINOv3 ViT-B/16 @288 (separate branch) | 288 | G2 gate failed (−0.0004) |

**Shipped:** `main_v5_C` = 6 full-data arm-C seeds (2026-2526), equal-weight mean of sigmoids, no TTA, K_infer 24, fp16.
Public LB 0.948, gold-58 0.9254. Its weakest findings on gold are synovitis ~0.81, Lat OA ~0.84 and PF OA ~0.88.
ACL and effusion reach ~0.98.

### 2.7 Kaggle export and inference (`kaggle_export.py`, `infer.py`)

- **Export:** a dataset holding the safetensors models plus `ensemble.json` (members, K_infer, a prior row,
  `wall_guard_hours` 8.25) and offline decoder wheels (pylibjpeg*, gdcm).
- **Notebook:** generated with the modules inlined. A test checks that the committed notebook is a fresh build.
  Kernels pin the docker image `sha256:37c64f7d`, because torch 2.11 broke the wheels.
- **Inference:**
  - re-runs the exact cache path for each study;
  - a CPU decode pool feeds one consumer thread per GPU, in fp16;
  - any study that fails gets the prior row, and the run exits with code 3 when more than 10 % of rows are prior;
  - the submission is validated against `sample_submission.csv` and written atomically.
- **Hang protection** (added after timeout 56837972): a watchdog thread (8.5 h hard limit, 30 min without progress)
  and heartbeat-bounded waits, falling back to `os._exit`. The output stays bit-identical.
- **Joint kernels** (team period, `qa/joint/`): each team's leg runs as a separate process and the outputs are
  rank-blended per finding. Leg failures fall back to the remaining legs. `--diag-encode` writes the failure code
  into the score, so the LB itself shows which leg failed. Scripts are zlib + base64 packed to stay under Kaggle's
  1 MB script cap.

### 2.8 Infrastructure

- GPU server (internal host): project `/raid/sroziewski/sandbox/rsna_kad`, data `/raid/sroziewski/data/rsna_kad`,
  cache `/raid/sroziewski/cache/rsna_kad`, Python `/raid/sroziewski/sandbox/bin/python`. Nothing is ever written
  to the home dir.
- `scripts/run_screen.sh <name> <cmd>` starts a detached `screen` with the project environment and writes its log
  to `logs/`. It redirects HOME, TMPDIR and HF_HOME into the project and records the git sha for bundle provenance.
- `scripts/remote.ps1` (Windows → server): deploy, token, preflight, start, status, logs, stop, test, sync
  (refuses when a drift check finds newer files on the server), drift, pull, run.
- `scripts/kaggle_cli.sh`: whoami, dataset-create / version / status, kernel-push / status / output, submit,
  submissions, raw.
- Tests: 26 modules, ~296 tests in the local mirror (`tests/`; the server suite is larger, 432 tests at AR L722), run on the server through `remote.ps1 test`. They use synthetic DICOM
  (2D, 3D, truncated, Philips rescale, signed Toshiba). The real vLLM run and the authenticated Kaggle calls are not
  covered.

---

<a id="team"></a>

## 3. Team and contributions

Team **"Roziewski & Jiang & Tikhomirov"**. Szymon (Simon) Roziewski (`sroziews`, owner of this repo) worked solo
until 10-14. Qile Jiang (`qilejiang76`) merged on **2026-10-14** and Andrey Tikhomirov (`andreytikhomirov`) on
**2026-10-15** (AR L1163, L1228). After the merge the team had one shared budget of 5 submissions a day. Team chat
was on Discord.

### 3.1 Szymon Roziewski

**Own pipeline (09-17 → 10-10)**, details in [Pipeline](#pipeline):
- Data fetcher for the 570 GB dataset, the preprocessing cache, site-grouped folds, and the gold-58 protocol.
- Report → label extraction: rules → Qwen3-32B-AWQ (2 prompts) → 4-voter merge (+ pilkwang public table; Gemma-3-27B was the clean-provenance alternative),
  then fold-local calibration.
- Model arms A (ConvNeXt-S 384), E (EffNetV2-S 256), C (ConvNeXt-V2-T FCMAE 384, 5 views). Pseudo-labels round 1
  (the biggest lever, +0.010 LB). Full-data retrains, ending in **main_v5_C** (6 Cfull seeds, LB 0.948, gold-58
  0.9254 for the 6-seed mean; 0.9253 from the end-to-end kernel run).
- Kaggle inference kernel with a watchdog against hangs, plus the export, parity and liveness tooling.
- About 25 rejected experiments, each with a pre-declared gate (see [Lessons](#lessons)).

**Public-model stacks (10-10 → 10-13):** main_v5_C rank-blended with public notebooks. OAI stack 0.953 → v2stack
0.955 → **v3stack** (heliosli R7 + nartaa Raptor 0.7, ours 0.3) **0.957**.

**Team infrastructure (10-14 → 10-22):**
- `qa/joint/build_joint.py` + `joint_template.py`: one kernel that runs several teams' legs as separate processes
  and rank-blends them per finding. It has failure fallbacks, `--require-all-legs`, `--diag-encode` (encodes which
  leg failed into the score) and `--total-h` (time budget). This built joint v2 (**0.959**, team best on 10-15),
  joint3 and joint3q3.
- Ran Andrey's OOF kernels on the GPU server (blurred MT-hires and R6/R7 OOF for 4,349 studies).
- Probes: ext-C, r6base, s6r, Opus-label C, leg blur (joint3blur 0.960, joint3blurall error, joint3blurall2 0.960), bpblur / bpsyn (0.961 ties), bprefit.

### 3.2 Qile Jiang

**Before the merge:** solo 0.957 (AR L1163-1166).

**Labels:** "Opus Blend" soft labels (`labels_opus_blend.csv`, in `Qile's labels and predictions.zip`). They come from
Claude Opus 5.5 run 3 times and averaged, with a product-of-experts confidence per cell, blended with soft
pseudo-labels; the blend weights were tuned on gold-58. Scored as labels on gold-58 they reach **0.9364**, against
0.8728 for our calibrated v1.1 *targets* (AR L1211; the raw v1.1 voter merge in [Data](#data) is 0.866). Retraining our C leg on them gave a tie (`qa/opus`).

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

### 3.3 Andrey Tikhomirov

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
bpblur and bpsyn. Both final picks are Andrey's (see [Finals](#finals)).

### 3.4 Merge history

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

### 3.5 Where each person's artefacts live

| Who | Local | Kaggle |
|---|---|---|
| Szymon | the whole repo; `qa/joint`, `qa/blur`, `qa/bblur`, `qa/bprefit`; `share/` | kernels `sroziews/rsna-kad-*` (joint, joint3, joint3q3, joint3blur*, bpblur, bpsyn, main-v5-c) |
| Qile | `Qile's labels and predictions.zip`, `qa/qile/` | notebook `qilejiang76/rsna-knee-ours-mtr1-mth4-olp5`; datasets above; kernel `rsna-kad-flat7` |
| Andrey | `hx48_ensemble.zip`, `qa/andrey/`, `.tmp/andrey_stacker5_zip/`, `.tmp/for_andrey/` (OOF we made for him), `files/` (curated labels) | kernels `rsna-knee-native-hx48`, `rsna-knee-j3-stack-blur`, `rsna-knee-flat7-blur`, `rsna-knee-flat4-syn-r6-s1` (+ `-stk6`, `-es2`), `rsna-knee-flat4-syn-r6-stk6-50`, `rsna-knee-mth-blur-oof`, `rsna-knee-r6-r7-blur-oof`. The S48 stacker JSON (`research/d_stacker_s48_6.json`) is in Andrey's own repo |

---

<a id="timeline"></a>

## 4. Timeline and submissions

The full day-by-day record is `docs/architecture.md` §4 (L244-1661). This page is the condensed version. LB means
the public leaderboard (3 decimals); gold means macro AUC on gold-58.

### 4.1 Public LB best over time

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

### 4.2 Phases

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

### 4.3 All submissions recorded in `architecture.md`

#### Szymon solo (09-19 → 10-13)

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

#### Team (10-15 → 10-22)

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

---

<a id="finals"></a>

## 5. Final ensemble

### 5.1 The two final picks

| Pick | Submission | Kernel (Andrey) | LB | Role |
|---|---|---|---|---|
| U8 | 57519548 | `rsna-knee-flat4-syn-r6-s1-stk6` v1 | 0.961 | upside: per Andrey, +10 LB positions over our other 0.961s at full precision |
| C3S | 57455318 | `rsna-knee-flat4-syn-r6-s1` v1 | 0.961 | hedge: flat 0.25 weights, few tuned parts, no gold search |

Both belong to Andrey's C3 lineage. No model Szymon trained is in either final: main_v5_C left the picks when U8
replaced B on 10-22.

### 5.2 What the members are

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

### 5.3 How they are combined

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

### 5.4 The predecessor B (final pick until 10-22)

`B j3-stack-blur` (57411322, 0.961, gold 0.9509) regrouped the joint3q3 members:
`0.28 rank(R7) + 0.72 rank(0.8 anchor + 0.2 stacker5)`, with blur on R7 and MTH. It was the only final candidate
containing our v5c leg and R7, and so the most different from the C3 family. The team traded that diversity for
U8's higher full-precision rank.

### 5.5 Why everything tied at 0.961

- The public LB covers ~30 % of 1,322 studies, and the SD of one submission is ~0.003 (forum and the leoprovorov
  write-up; AR L1347, L1597).
- Gold-58 transfers to LB at a slope of 0.24, so +0.001 on gold is about +0.0003 on LB (AR L1536-1539).
- The adoption bar was therefore **≥ 0.962**, and nothing reached it. Ties could only be ordered by the UI order
  and Andrey's estimate from leaderboard positions.
- Kaggle staff warned that public blends tend to drop on the private LB. That is the argument for keeping the
  less-tuned C3S as the hedge.

---

<a id="lessons"></a>

## 6. What worked, what didn't, and how we decided

### 6.1 What moved the score

| Lever | Gain | Who |
|---|---|---|
| LLM report labels instead of rules (Qwen3-32B, 2 prompts) | gold labels 0.816 → 0.852 | Szymon |
| Better labels plus a stronger backbone at 384 px (MVP → arm A; the MVP recipe on LLM labels already scored 0.913) | LB 0.865 → 0.931 | Szymon |
| Diverse label voters (public table / Gemma-3) | +0.012-0.015 label AUC | Szymon |
| **Pseudo-labels round 1** (0.5 label + 0.5 teacher OOF) | gold +0.0135, **LB +0.010** (0.933 → 0.943) | Szymon |
| Full-data retrains, C-only ensemble | LB 0.943 → 0.948 | Szymon |
| Public models (R7, Raptor) with our C leg added (plain R7 alone: 0.956) | LB 0.948 → 0.957 | Szymon |
| Team merge: rank-blending independent model families | LB 0.957 → 0.959 | Szymon + Qile + Andrey |
| **Test-time Gaussian blur** (0.288 mm R6/R7, 0.8 px hires) | R7 +0.0101 gold; LB 0.959 → 0.960/0.961 | Andrey |
| Out-of-fold stackers (stacker5, S36, S48) | LB 0.961; U8 the best 0.961 at full precision, per Andrey | Andrey |

### 6.2 Dead ends (tie or loss on the pre-declared gate)

- **Labels:** "strict" prompt, confident-learning relabelling (only 0.10 % of cells contradicted), Claude Sonnet
  voter, silentq, glmgrade (GLM grading of "mild"), agreement masking, Opus-label retrain, stale-teacher removal,
  t70 heavier teacher, multi-source teachers, pseudo round 2.
- **Models:**
  - CoAtNet arm B (slow, underfit), ConvNeXt-V2-base (tie at 3x the cost), per-label attention and transformer heads;
  - Pipeline B (DINOv3): diversity disappeared once both families were pseudo-labelled (ρ 0.77 → 0.90);
  - max-pool MIL, nnPU loss, DANN against hospital (**hurt**, −0.0126), pairwise loss, MRI intensity augmentation;
  - synovitis second-stage classifier.
- **Inference:** z-TTA (+0.0005), crop TTA (LB tie), rotation TTA (screened on gold; only crop went to LB), blur on our own C leg (no effect), learned stackers
  over our own 10 models (all lost; they only helped later on diverse team members, out of fold).
- **External data:** OAI-trained C (LB tie), MRNet / KneeMRI C (LB tie), GLM-5.3 vision labeller (0.66-0.73 gold).
- **Ensembles:** stk (re-teaching on the public stack lost diversity), r6base, P+A without V, equal thirds, s6r,
  bpblur, bpsyn, bprefit, the U-series, the C3SE expert: all ties at the noise level.
- **Merges declined:** candidates whose gold-58 predictions did not raise ours, or who offered no own models.

### 6.3 Decision rules (how they evolved)

1. **09-21 (`qa/arbiter`, AR §4.2):** accept at ≥ +0.010 with P ≥ 0.95. 0.005-0.010 needs every metric to agree.
   Below 0.005 is a tie. Gold-58 is a **veto**. The public LB (~390 studies) resolves only ~0.02, so it was never
   used for ranking.
2. **09-26 (AR §4.5):** paired differences only. Accept at ≥ +0.004 on gold with no drop below −0.009, plus one
   paired LB probe.
3. **10-03 on:** a pre-declared test on 3 full-data seeds against twins, paired bootstrap on gold-58: accept iff
   Δ ≥ +0.004 at P ≥ 0.95. Per-label deltas were reported but not acted on.
4. **10-15 / 10-16:** Andrey showed that gold under-rates models that help on normal knees. From then on, the judges for
   extra-knee legs treated gold as information only, with a floor of −0.004 (AR L1316-1317).
5. **10-20 on:** LB-vs-gold slope 0.24 and an LB SD of ~0.003, so a new pick needed **LB ≥ 0.962**. Variants were
   pre-registered with their own rules (e.g. C3SE: "finals only if above C3S in the UI order").

### 6.4 Lessons

- **Labels mattered more than architecture.** The biggest jumps came from report labels and pseudo-labels. More arms
  of the same kind saturated (+0.0077, +0.0049, +0.0007).
- **Diversity has to be measured.** It disappears once members are trained on the same teacher: Pipeline B,
  stk, and the Opus-label C all drifted towards the existing members.
- **Gold-58 is small and biased** (no normal knees, enriched for positives). Use it as a veto with paired bootstrap,
  never for weight tuning. Weights tuned on gold are circular.
- **The Kaggle time budget is a correctness issue.** A leg timed out silently in joint3 v1 (confirmed by diag
  code 3). joint3blurall's ERROR was attributed to the budget when joint3blurall2 passed with an 8.8 h budget (leg A unblurred,
  `--diag-encode`).
  The s6r ERROR was most likely a timeout too, but that is not proven (AR L1495-1503, L1587). `--diag-encode` (the failure code shows up in the LB score) was the only way to see it on
  the hidden test. `--require-all-legs` reports a timeout as a crash.
- **Pin the docker image.** Kaggle's switch to torch 2.11 broke our decoder wheels.
- **Pre-declare gates before seeing results.** Across solo and team work, this kept ~40 tie-level experiments from
  being adopted on noise.
- **Last-epoch selection.** Best-epoch selection is ~+0.008 optimistic.

### 6.5 Gaps left

- The LB scores of 57193974 (v4pf) and 57298855 (blend5) are not recorded.
- Fit details of the S48 stacker exist only in Andrey's repo (`research/d_stacker_s48_6.json`).
- Never confirmed: whether gold-58 was kept out of Qile's FULL retrains.
- About 20 qa/ directories exist only on the server (see [Inventory](#inventory)).
- Two infra bugs were never fixed: a timed-out leg's grandchild processes survive, and an R7 failure inside leg V
  does not show up in the diag codes.

---

<a id="inventory"></a>

## 7. Scripts and experiment inventory

### 7.1 `src/rsna_kad/`

| Module | Purpose | Key entry points |
|---|---|---|
| `constants.py` | label names / order, under-reported keys, slots, server paths | `LABELS`, `LABEL_KEYS`, `MAIN_SLOTS` |
| `config.py` | YAML → dataclasses (unknown keys raise) | `load_config`, `TrainConfig` |
| `dicom.py` | pydicom header / slice reading, rescale, geometry | `read_header`, `read_slice`, `series_geometry` |
| `series.py` | series → slot routing, ranking, FS → T1 fallback | `assign_slots` |
| `preprocess.py` | fill, mm crop, z-slab, resize, p1/p99 window | `series_to_volume` |
| `laterality.py` | side from tag / description / geometry | `estimate_sides` |
| `cache.py` | resumable parallel cache build, index, verify | `build_cache`, `load_study`, `verify_cache` |
| `cv.py` | site-grouped balanced folds | `make_folds`, `check_folds` |
| `data.py` | StudyDataset, slice sampling, augmentation, view crops | `StudyDataset` |
| `mri_aug.py` | optional bias / gamma / blur / noise | `mri_augment` |
| `tta_views.py` | crop / rotation eval views (qa/tta2) | `ViewTTADataset` |
| `model.py` | KneeMIL + head variants, bundle save / load | `build_model`, `load_bundle` |
| `train.py` | fold / full-data training, EMA, mixup, OOF | `train_fold`, `assemble_oof` |
| `metrics.py` | AUC, gold report, arbiter subsets, paired bootstrap | `gold_report`, `paired_delta` |
| `runguard.py` | refuse scoring / exporting live or smoke runs | `assert_not_live` |
| `infer.py` | raw-DICOM test inference, multi-GPU, guards | `notebook_main`, `write_submission` |
| `kaggle_export.py` | export bundle, wheels, notebook generation, Kaggle wrappers | `export_bundle`, `build_notebook` |
| `labels/schema.py` | output JSON schema + parsing | `JSON_SCHEMA`, `parse_output` |
| `labels/rules.py` | 9-language rule voter | `extract_rules` |
| `labels/prompts.py` | V1 / V2 prompts, rule wordings, few-shots | `build_messages`, `assert_no_gold` |
| `labels/llm.py` | vLLM runner (resumable) | `run_extractor` |
| `labels/merge.py` | voter merge, evidence check, extra voters | `merge_votes`, `add_voter` |
| `labels/targets.py` | EB calibration + PAVA, targets / weights | `build_fold_targets` |
| `labels/pseudo.py` | stage-9 and v3 teacher mixing | `mix_with_teacher` |
| `labels/public.py` | public label table adapter | `load_public_table` |

### 7.2 `scripts/`

| Script | Purpose |
|---|---|
| `kaggle_fetch.py` | resumable fetch / extract / verify of the 570 GB archive (stdlib only; hole-punching keeps peak disk ≈ dataset size) |
| `remote.ps1` | Windows → server driver: deploy, token, preflight, start, status, logs, stop, test, sync, drift, pull, run |
| `run_screen.sh` | start a detached `screen` job with the project environment and a log |
| `kaggle_cli.sh` | Kaggle CLI with the project token: dataset / kernel / submit / submissions |
| `build_cache.py` | build / verify / export-index of the preprocessing cache |
| `make_folds.py` | write `folds_v1.csv` once (refuses to overwrite without `--force`) |
| `extract_labels_rules.py` | rule-voter table + gold report |
| `extract_labels_llm.py` | LLM draft / run / merge (`.venv-llm`, GPU 0) |
| `make_labels_v1_1.py` | add a 4th voter |
| `make_targets.py` | per-fold soft targets |
| `make_pseudo_targets.py` | stage-9 leakage-proof pseudo sources |
| `targets_v1_1_chain.sh` | GPU chain for 4th-voter candidates on the 481-report subset |
| `train.py` | train one fold or `--fold full`, `--seed` |
| `predict_oof.py` | assemble / recompute OOF + metrics |
| `compare_runs.py`, `compare_ensemble.py`, `evaluate_runs.py` | paired bootstrap comparisons across runs, blends and arbiter subsets |
| `export_kaggle.py` | build `export/<name>`, regenerate the notebook, push dataset / kernel |
| `bench_infer.py` | parity, precision, K_infer and timing measurements |

Top level: `pipelineA.py` (single-file reproduction of main_v5_C: build-cache, make-folds, make-targets, train,
predict, score), `pipelineA_core.py` (excerpt of the core ideas, for discussion), `notebooks/kaggle_inference.py`
(generated kernel). `configs/` holds 60 YAMLs: cache, folds, targets, LLM, and training arms MVP / E / A / B / C / D /
N plus stage-9 sources.

### 7.3 `qa/` experiment directories

| Dir | Purpose | Result |
|---|---|---|
| `stage1` | cache v1 + folds QA | accepted: 24,386 series, 98.2 GiB |
| `stage3` | MVP training + leakage audit | accepted: OOF 0.846, gold 0.847, no leak |
| `stage4` | first Kaggle kernel: parity, precision, timing | LB 0.865, bit-exact parity |
| `stage5` | rules / LLM labels + targets QA | accepted; 8 % of spot-checked cells wrong |
| `stage7` / `stage8` | efficiency / ensemble kernel outputs | LB 0.913 / 0.929 |
| `stage8b` | arm D crop geometry | 90 mm coronal crop cuts MCL in 78 %, so 100 mm |
| `stage9` | confident learning + pseudo-label design | denoising rejected (0.10 % of cells) |
| `stage11` | per-label attention / transformer heads | closed: tie |
| `arbiter` | which offline metric to trust | the decision rule (§6.3) |
| `audit_2026-09-25` | six-role project audit | arms saturated; labels are the bottleneck |
| `llm_ab` | LLM quantisation / prompt A/B (481 studies) | keep AWQ + base prompt |
| `claude_labels` | Claude Sonnet as a 4th voter | marginal +0.0043, full pass not run |
| `competitor` | review of a public 0.942 notebook + public label table | public table adopted as voter (v1.1) |
| `ensemble` | blend analysis + timing | A + eff +0.0077 OOF |
| `pseudo_v3` | **pseudo-labels round 1** | main_v4 gold +0.0135, LB 0.943 |
| `full` | full-data retrains + paired judge | main_v5_C LB 0.948 |
| `hpe` | hyper-parameter-diverse seed ensemble | tie |
| `main_v5` | arm-balanced export fix | v5_C 0.948, v5_bal 0.945 |
| `main_v5c12` / `_bweak` / `_crop` / `_mpnnpu` / `_silentq` / `main_t70` | release dry runs of LB probes | all 0.944-0.948 ties |
| `mriaug` | MRI intensity augmentation | no gain |
| `ztta` / `tta2` | z-offset / crop + rotation TTA | dropped / LB tie |
| `stacking` | learned stackers over our 10 models | all 5 lose |
| `sota_2026-09` | literature scan + Pipeline B design | DINOv3 built, G2 failed |
| `release_v7` | merged A + B release dry run | unused after G2 failed |
| `hardening` | hang-protected notebook verification | bit-identical |
| `eng` | engineering changelog | server git repo, export / liveness guards |
| `oai` | OAI MOAKS knees as training data | LB tie 0.955, closed |
| `v2stack` / `v3stack` | public stacks + our C leg | LB 0.955 / **0.957** |
| `lbgold` | LB vs gold-58 plot | Spearman 0.95 over 22 submissions (the 0.24 slope is a later 54-submission analysis) |
| `agree` | mask cells where voters contradict | stop (−0.0011) |
| `opus` | C leg on Qile's Opus labels | tie / reject |
| `qile` | integrate Qile's leg, faster notebook parity | joint v2 0.959 |
| `andrey` | integrate Andrey's hx48, OOF kernels on our server | joint3 gold 0.9462; LB 0.959 was the V/P fallback (leg A timed out); all legs ran in joint3q3 (0.959) |
| `joint` | joint kernel template (legs, fallbacks, diag-encode, total-h) | the team kernel infrastructure |
| `ext` | C leg with MRNet / KneeMRI knees | LB 0.959 tie |
| `r6base` | public 0.958 recipe as V base | rejected (−0.0017) |
| `s6r` | nartaa S6 trainer on our labels | rejected; probe errored |
| `glm_vision` | GLM-5.3-Flash as image labeller | rejected (0.66 gold) |
| `discussion_2026-10-18` | forum / kernel scan | nothing above 0.958 public |
| `blur` | inference blur on C, legs P / A, R7 | no effect on C; joint3blur LB 0.960 |
| `bblur` | bpblur / bpsyn kernels | LB 0.961 ties |
| `bprefit` | stacker5 refit on blurred OOF | rejected (+0.0004, P 0.66) |

Server-only qa dirs, with verdicts from `architecture.md`: `main_v4`, `maxpool`, `nnpu`, `silentq`, `silentq_syn`,
`glmgrade`, `glm_text`, `q1test`, `dann`, `synov`, `t70`, `mt`, `stk`, `oaistack`, `reader`, `v4pf`, `pa`,
`pipeline_b`, `discussion_2026-10-11/14/15`.

### 7.4 Other files at the repo root

| File | What |
|---|---|
| `Qile's labels and predictions.zip` | Qile's Opus Blend labels, member OOF / gold predictions, AUC table |
| `hx48_ensemble.zip` | Andrey's hx48 kernel (notebook, metadata, README) |
| `merge_share_2026-10-15_gold58.zip` | gold-58 predictions from a merge candidate |
| `invisible_ensemble_merge_proposal.md` | a declined merge offer from another team |
| `latest_gold.csv` | Andrey's hx48 predictions on gold-58 |
| `share/` | joint kernel code + our gold-58 predictions, sent to merge partners |
| `gpt_prompt.md`, `gpt_message.txt`, `gpt_followup.txt` | external-LLM review prompts about our pipeline (0.948 era) |
| `train_files.zip` | 0-byte file, not an archive |
