# 1. Data

Sources: `docs/data_analysis.md` (DA), `docs/research.md` (RS), `docs/architecture.md` (AR, cited as `AR L<n>`),
`analysis/out/*`. Raw numbers come from `analysis/eda.py`; the fact dump is `analysis/out/report_facts.txt`.

## 1.1 Competition data

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

## 1.2 Gold-58 (the only image-graded labels we had)

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

## 1.3 Folds

`data/folds/folds_v1.csv`: 5 folds of 879-883 studies, grouped by `site_key`, with 11-12 gold studies per fold. Built
by a seeded greedy balancer (`cv.make_folds(method="balanced")`), because sklearn `StratifiedGroupKFold` failed the size
and gold criteria for all 200 seeds tried. Full-data runs train on all 4,349 non-gold studies and are scored on
gold only. Qile's g1 models use folds_v1 too (AR L1638). The heliosli R6/R7 OOF uses hash folds (sha1(uid) % 4).

## 1.4 Label versions

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

## 1.5 External data

| Dataset | Used by | Outcome |
|---|---|---|
| OAI (2,439 MOAKS knees, 75 GB, NDA package) | Szymon (`qa/oai`) | own OAI-trained C leg: LB tie 0.955, closed. OAI is also inside the public nartaa Raptor weights and one of Qile's legs |
| MRNet (Stanford AIMI, 1,250 exams) | Andrey (hx model), Szymon (`qa/ext`) | in hx48; ext-C LB 0.959 tie |
| KneeMRI (Zenodo, CC BY-NC-ND 4.0, 745 series) | Andrey, Szymon | as MRNet |
| fastMRI | heliosli R7 public model (auxiliary) | inside R7 |

Host rulings: external data is OK when it needs only registration and a licence acknowledgement (post 743416,
10-14). Hosted LLMs are allowed for labelling (post 733965). External data has to be disclosed in the write-up.

### What the competition write-up must disclose

The external data and weights inside the final picks:
- **public weights:** heliosli R6 (and R7, which used fastMRI as an auxiliary, in B); nartaa Raptor (trained with
  OAI) in earlier picks;
- **Qile's models:** one leg uses OAI;
- **Andrey's hx model:** MRNet (Stanford AIMI) and KneeMRI (Zenodo, CC BY-NC-ND 4.0);
- **hosted LLMs used for labels:** Claude Opus (Qile's labels), the pilkwang public LLM table, Qwen3-32B (ours,
  self-hosted). Gemma-3-27B was tried but is not in the final labels.

## 1.6 Where the data lives

- Server copy: `/raid/sroziewski/data/rsna_kad/` (CRC-verified), fetched by `scripts/kaggle_fetch.py`
  (range-resumable, extracts while downloading; see `README.md`).
- Preprocessing cache v1 (server): 24,386 series, 98.2 GiB, uint8 (N, 384, 384), 130 mm crop, built in 265 s.
- Repo: `data/labels/`, `data/targets/v0|v1/`, `data/folds/`, `data/fewshot/`, `data/oof_train_armC_5fold.csv`;
  EDA tables in `analysis/out/` (~40 CSV/JSON plus `headers.parquet`, one row per DICOM file).
- `train_files.zip` in the repo root is a **0-byte file**, not an archive.
