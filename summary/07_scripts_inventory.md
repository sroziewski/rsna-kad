# 7. Scripts and experiment inventory

## 7.1 `src/rsna_kad/`

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

## 7.2 `scripts/`

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

## 7.3 `qa/` experiment directories

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

## 7.4 Other files at the repo root

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
