# RSNA Knee Abnormality Detection

Team **Roziewski & Jiang & Tikhomirov**'s solution to the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle
competition (2026-09-17 → 2026-10-22): predict 12 abnormalities from knee MRI studies, scored by macro ROC AUC.

| | |
|---|---|
| **Final place** | **85 of 6,425 teams**, silver zone |
| **Public LB** | 0.961 |
| **Final picks** | U8 `57519548` + C3S `57455318` (Andrey's rank blends of R6, MT-hires, OL-plane and hx48 with out-of-fold stackers) |
| **Team** | Szymon (Simon) Roziewski · Qile Jiang · Andrey Tikhomirov |

**Start here:** [`summary/RSNA_KAD_team_summary.md`](summary/RSNA_KAD_team_summary.md) is the full write-up as one
file (data, pipeline, each teammate's work, every submission, the final ensemble, lessons). The same content split by
topic is in [`summary/`](summary/README.md). The day-by-day engineering log is
[`docs/architecture.md`](docs/architecture.md).

## The problem in one paragraph

Only **58 of 4,407** training studies carry image-graded labels. The other 4,349 have to be labelled from free-text
radiology reports in 9 languages, and those reports under-report: synovitis is affirmed in 12.5 % of reports but
present in 46.6 % of graded studies. The data is ~570 GB of uncompressed DICOM from about 16-19 sites with stripped
site tags. The hidden test set is ~1,322 studies; the public LB shows ~30 % of it, rounded to 3 decimals.

## What this repository contains

Our own pipeline (Szymon), which scored **0.948** alone, plus the team tooling that combined it with Qile's and
Andrey's models:

```
reports ─► rule voter + Qwen3-32B (2 prompts) + public-table voter ─► calibrated soft targets
        ─► pseudo-labels (0.5 label + 0.5 teacher OOF)
DICOM   ─► cached uint8 volumes per series ─► KneeMIL (ConvNeXt-V2-T, 2.5D, 5 views, gated-attention MIL)
        ─► 6 full-data seeds ─► Kaggle kernel (offline wheels, watchdog, multi-GPU)
team    ─► joint kernel: each team's leg in its own process, per-finding rank blend, fallbacks, diag codes
```

| Path | What it is |
|---|---|
| [`src/rsna_kad/`](src/rsna_kad) | the package: DICOM reading, series routing, preprocessing cache, folds, label extraction (`labels/`), model, training, metrics, inference, Kaggle export |
| [`scripts/`](scripts) | CLIs: data fetch, cache build, folds, label extraction, targets, training, OOF, comparisons, export, server drivers |
| [`configs/`](configs) | 60 YAMLs: cache, folds, targets, LLM extractors, training arms (MVP, E, A, B, C, D, N) |
| [`tests/`](tests) | pytest suite with synthetic DICOM fixtures (run on the server) |
| [`pipelineA.py`](pipelineA.py) | single-file reproduction of the shipped `main_v5_C` recipe |
| [`notebooks/`](notebooks) | the generated Kaggle inference kernel |
| [`qa/`](qa) | one directory per experiment or probe, each with its scripts and, where present, a `report.md` (~50 dirs, inventory in the summary) |
| [`docs/`](docs) | `architecture.md` (design + full log), `data_analysis.md` (EDA), `research.md` (rules, literature), `data_fetch.md` |
| [`summary/`](summary) | the final write-up |
| [`data/`](data) | small tracked tables: folds, few-shot examples, label stats, OOF |

Model weights, the preprocessing cache, raw DICOM, the full per-study label tables and run outputs are **not** in
git; they live on the team's GPU server (see `.gitignore`).

## Running it

Requires Python ≥ 3.11 and a CUDA GPU. Install a CUDA build of `torch` first, then `pip install -e .`. The paths
default to the team server; override them with `RSNA_KAD_DATA`, `RSNA_KAD_CACHE` and `HF_HOME` (see
`src/rsna_kad/constants.py`). Run the tests with plain `pytest`.

The full recipe is a chain of steps. Each depends on the outputs of the one before, which are not in git:

| Step | Entry point |
|---|---|
| 1. Download the 570 GB dataset (resumable, extracts in place) | `scripts/kaggle_fetch.py`, see [`docs/data_fetch.md`](docs/data_fetch.md) |
| 2. EDA tables (`analysis/out/`, needed by the fold builder) | `python analysis/eda.py` |
| 3. Preprocessing cache | `python scripts/build_cache.py build --config configs/cache_v1.yaml --split train` |
| 4. Folds | already tracked as `data/folds/folds_v1.csv` (md5 pinned in `tests/test_cv.py`); `scripts/make_folds.py --force` rebuilds it |
| 5. Report labels: rules + Qwen3-32B-AWQ on vLLM (own venv `.venv-llm`, ~24 GB GPU) | `scripts/extract_labels_rules.py`, `scripts/extract_labels_llm.py`, `scripts/make_labels_v1_1.py` |
| 6. Soft targets | `scripts/make_targets.py` |
| 7. Teacher models + OOF, then pseudo-label targets | `scripts/train.py` per fold, `scripts/predict_oof.py`, `qa/pseudo_v3/`, `qa/full/make_targets_full.py` |
| 8. Full-data training of the shipped recipe | `python scripts/train.py --config configs/main_armc_pseudo_full.yaml --fold full --seed 2026` (6 seeds) |
| 9. Kaggle bundle + inference notebook | `scripts/export_kaggle.py` |

[`pipelineA.py`](pipelineA.py) runs the same recipe from one file, and the
[Pipeline section](summary/02_pipeline.md) of the write-up explains each step.

## Results at a glance

| Stage | Public LB |
|---|---|
| MVP EfficientNetV2-S 256 px, rule labels | 0.865 |
| ConvNeXt-S 384 px on LLM labels | 0.931 |
| Pseudo-labels round 1 | 0.943 |
| `main_v5_C`: 6 full-data ConvNeXt-V2-T seeds (our best solo) | 0.948 |
| our leg rank-blended with public models | 0.957 |
| team merge (joint kernel with Qile's models) | 0.959 |
| test-time blur + out-of-fold stackers (Andrey) | **0.961** |

What worked, what didn't (about 40 tie-level experiments across solo and team work, all under pre-declared gates)
and why are in the
[Lessons section](summary/06_lessons.md) of the write-up.

## Credits

- **Szymon Roziewski:** data pipeline, report labelling, KneeMIL models, Kaggle inference, joint team kernel.
- **Qile Jiang:** Opus Blend soft labels (the best label table on gold-58); ResNet-50 / ConvNeXt-T models MT-r50,
  MT-hires and OL-plane; full-data retrains, a faster inference notebook, team blends flat5-7.
- **Andrey Tikhomirov:** native-DICOM CoAtNet MIL (hx48), test-time blur sweep, out-of-fold stackers, the final picks.

Public models by heliosli (R6 / R7) and nartaa (Raptor, S6), and the pilkwang public label table, were used as
ensemble members or label voters; external data (OAI, MRNet, KneeMRI, fastMRI) is listed in the
[Data section](summary/01_data.md).

Competition data belongs to RSNA and is used under the MIRA licence, which forbids redistributing it. This repository
contains no DICOM, no report text and no gold labels: the few-shot examples (`data/fewshot/*.json`), the labelling
batches and prompts in `qa/claude_labels/` and teammates' raw files are kept out of git. To rerun the report
labelling, rebuild the few-shot files from your own copy of the competition data.
