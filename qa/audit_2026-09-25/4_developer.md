# Developer audit - engineering, reproducibility, release readiness (2026-09-25, deadline 2026-10-22)

Scope: `src/rsna_kad/**`, `scripts/*`, `configs/*`, `tests/*`, `notebooks/`, `pyproject.toml`, plus the server working copy `/raid/sroziewski/sandbox/rsna_kad` (read-only: md5 comparison, `pytest --collect-only`, export and run listings, HF cache refs). Nothing was modified outside this file. Live screens at audit time: `armD`, `armn_pick`, `armn_triage` (GPU 0 at 99 %).

## 0. State of the tree (no VCS)

- **Not a git repository, locally or on the server.** Every bundle carries `git_sha: ""`: `train.py:396-397` reads `RSNA_KAD_GIT_SHA`, which nothing sets (checked in `export/main_v3/models/*.json`). No bundle, export or LB score can be tied to a source state after the fact.
- **The two copies have already drifted.** `src/ scripts/ configs/ tests/` are md5-identical, but `notebooks/kaggle_inference.py` is not: local = digest `cd6204d52ecffe59` (built 09-21, no per-label heads), server = `ed438d7ae2011b1c` (regenerated after `model.py` gained `head`/`SLOT_PRIOR`). Locally `test_export.py::test_committed_notebook_equals_fresh_build` would fail. `docs/` exists only locally; `qa/stage10`, `qa/stage6_kernel` only on the server. `remote.ps1 sync` pushes one way, `pull` pulls `data/`; there is no single authoritative tree.
- Production steps live under `qa/`: `scripts/targets_v1_1_chain.sh` calls `qa/llm_ab/run_subset.py`; arm-N training runs from `qa/stage11/full_chain.sh`. The gemma3 full-corpus extraction and the `make_labels_v1_1`/`make_targets` commands that produced `data/targets/v1_1_gemma3` are in no script.

## 1. Correctness risks the tests would not catch

Submission path (`infer.py`, generated notebook):
1. **Hard failure modes lose the whole submission.** A consumer exception (e.g. CUDA OOM on one study) is re-raised after the loop (`infer.py:543-545`), so no `submission.csv` gets written. `pending.popleft().get()` (`infer.py:516`) has no timeout. One study whose decode hangs (gdcm/openjpeg on a malformed J2K) blocks the run until Kaggle kills it at 9 h, because the wall guard (`infer.py:529`) only runs after a study comes back. The design (§2.7 "never a crash") holds for exceptions in workers, but not for these two cases.
2. **The wall guard does not cover setup time.** The deadline counts from the start of `notebook_main` (`infer.py:710`), which excludes the interpreter and torch/timm import and the wheel install in the prelude. The 45 min margin (8.25 h) absorbs this today. Re-measure before adding arms D/N: 15 members is about 2.8 h projected, and 25 members at 5 views is untested.
3. **Preprocessing comes from the first member only** (`infer.py:477-480`). Every member currently uses cache v1. A future cache-v2 member (F3 p1/p99 fix) would silently get v1 volumes. Assert that all members share `params_sha256(cache)`.
4. **A silent prior flood is possible.** If the decoder wheels fail to install (`PRELUDE`, warning only) and the hidden test is compressed, every slice returns `None`, every slot goes missing, and every row becomes the prior. The file is still valid but scores about 0.5. Nothing counts prior rows against a threshold (`infer.py:569-570` only logs them).
5. Member weighting is 1.0 per *member* (`kaggle_export.py:133`), so weights are equal per arm only while every arm ships 5 folds. `scripts/export_kaggle.py:64-70` **skips a missing fold with a print and continues**, so an arm with 4 folds silently gets 4/14 of the weight instead of 1/3.
6. **OOF and shipped predictions differ.** OOF is scored without TTA, in bf16 (`train.py:356`); the kernel uses z-offset TTA and fp16. Only main_v2 has a recorded kernel-vs-local diff (2.1e-4). main_v3 (5-view ConvNeXt-V2) has no fp16/fp32 or kernel-vs-local record in docs.
7. Checked and correct: slot and fallback resolution match the cache (`infer.py:132-174` vs `cache.py:258-308`, same `assign_slots` keys: `n_src` equals `len(headers)`), row order and NaN/range checks (`infer.py:577-612`), and view crops (a guarded `load_member(views=True)` plus a parity test). Laterality is computed but no model consumes it, so there is no flip risk. Minor issues: only `PixelSpacing[0]` is used (`dicom.py:78`, anisotropic pixels would distort the crop), and a non-integer `Fluid_Sensitive` in `test_series.csv` silently drops the series (`infer.py:118`), whereas training would crash (`cache.py:140`).

Training/evaluation path:
8. **P0: `export_kaggle.py` can ship a half-trained fold.** With `select_epoch: last`, `bundle.json` and `model.safetensors` are rewritten after every epoch (`train.py:374-378`). `export_kaggle.py:67` accepts any fold dir that has `bundle.json`. It ignores `last.pt` and `metrics.epoch` and does not check whether the screen is alive. This is the "never score a live chain" trap moved onto the release path, and arms D and N are live right now.
9. Scoring tools have the same gap. `compare_ensemble.py:44-50` globs `fold*/oof.parquet` and averages duplicate IDs (`groupby().mean()`). `predict_oof.py`/`assemble_oof` and `evaluate_runs.py` have no liveness guard; only `qa/stage8b/compare.sh` has one. `predict_oof --recompute --tta` overwrites the training OOF in place.
10. **Target-version bookkeeping.** `data/targets/v1_1_gemma3/manifest.json` records `"version": "v1_1"` and `configs/targets_v1_1.yaml`, which cannot be told apart from the public-table `v1_1`, and `labels_path()` trusts this manifest. `data/targets/v2` has no manifest, so strict AUC is skipped for stage-9 runs. `compare_ensemble.py` defaults to `--targets v1`. That is right as a common yardstick (§4.2), but nothing stops comparing v1_1_gemma3 arms (D, N) on their own weak-AUC.
11. Fold and gold leakage: clean in the code I read. Gold is removed from targets and asserted out of train (`train.py:263-266`), calibration is fold-local, and `assert_no_gold` guards few-shots. The public `v1_1` table (hosted-API provenance) must not reach the final entry. Arms D and N use `v1_1_gemma3`, which is clean.

## 2. Reproducibility - can a stranger rebuild the final model?

No. What exists: seeded folds with a pinned md5, seeded per-item augmentation RNG (`data.py:182-183`), `torch.manual_seed(seed+fold)` (`train.py:251`), config copied into every bundle, and fingerprinted weights. What is missing:
- **Environment.** `pyproject.toml:9-19` has no version pins and there is no lock file. Training env: torch 2.9.1+cu128, timm 1.0.29, pydicom 3.0.2, numpy 2.3.4, pandas 2.3.3, cv2 5.0.0, sklearn 1.7.2. Kaggle image: torch 2.10, and its timm/pydicom are unpinned by us. **timm is imported from the Kaggle image, not shipped.** A timm model-definition change would abort at the strict-key/fingerprint check (loud, not silent). LLM venv: vLLM 0.29.0, torch 2.13, transformers 5.17, xgrammar 0.2.7. There is no freeze of either venv.
- **External weights.** The HF cache revisions currently match HF `main`. Record them, since nothing in the code pins them: convnext_small `14a8373`, convnextv2_tiny `388f573`, effv2_s `ea9abc1`, Qwen3-32B-AWQ `0499c3a`, gemma-3-27b-it-w4a16 `2b53755`. Gemma is stored outside `hub/` (`download_dir=HF_HOME`), and `targets_v1_1_chain.sh:141` deletes LLM weights after use.
- **Determinism.** Nothing sets `cudnn.deterministic` or `use_deterministic_algorithms`. `torch.compile` is on for all main arms. A resume does not restore the torch RNG, reseeds the shuffle generator from scratch (`train.py:277`, so epoch e after a resume replays epoch 0's order) and reseeds mixup (`train.py:309`). A resumed fold is therefore not the run it replaced. vLLM batched greedy decoding is not bit-reproducible. Ship the label tables and do not promise to regenerate them bit-for-bit.
- **Pipeline order is documented only as prose in §4.** It should be one script: fetch -> `build_cache.py` -> `extract_labels_rules.py` -> `extract_labels_llm.py` (x2 variants + gemma) -> `make_labels_v1_1.py` -> `make_targets.py` -> `make_folds.py` (or the pinned CSV) -> `train.py` x5 per arm -> `predict_oof.py` -> `export_kaggle.py`. The §4 stage-10 `docs/reproduce.md` and `export_kaggle.py --verify` do not exist: `verify` today only CPU-loads members (`kaggle_export.py:130-132`).
- Hard-coded server paths: `constants.py:42-44` defaults, `run_screen.sh:19,31`, `kaggle_cli.sh:35`, `targets_v1_1_chain.sh:18,21`, `bench_infer.py:240-242`, `remote.ps1:30-35`. The env overrides exist, but the shell scripts `cd` into `/raid`.

## 3. Technical debt / dead code

- 36 configs, of which about 20 are superseded: `mvp_*`, `smoke_*`, `main_smoke_384`, `main_coatnet2_384` (aborted), `main_convnext_s_384` (12 ep, replaced by the 8-epoch schedule), `armc_sweep_c1-4`, `armd_sweep_d0-d3`/`d1g`, `stage9_pseudo_*` and `main_armA8_v2` (not bought), `main_armc_v1_1_f0`, and `llm_qwen3_32b_{bf16,fp8}` and `llm_mistral_*` (A/B only). 28 run dirs, most of them sweeps or smoke runs.
- Additive growth: `model.py` has three heads plus `SLOT_PRIOR`/`NARROW_VIEW_KEYS`. `ModelConfig.crop_mm` duplicates `CacheConfig.crop_mm`: the prior uses the model copy (`model.py:190`), the views use the cache copy (`data.py:102`), and both are 130 today. `load_bundle` (`model.py:317-343`) and `load_member` (`infer.py:236-272`) duplicate the strict-load and fingerprint logic. `InferConfig` is dead (only inlined). `iter_studies` is unused.
- `qa/competitor/kernels/*/src.py` (2 x 5,191 lines) are third-party code. Exclude them from any release.
- The weights dataset is declared `CC0-1.0` (`kaggle_export.py:157`). That is wrong once NC-derived weights are inside (see section 4).

## 4. Release readiness (winners' obligations)

Licences checked on the HF model cards, 2026-09-25:

| Component | Licence | Used by | Verdict |
|---|---|---|---|
| `convnext_small.fb_in22k_ft_in1k_384` | Apache-2.0 | arm A | OK |
| `tf_efficientnetv2_s.in21k_ft_in1k` | Apache-2.0 | eff, main_v2/v3 | OK |
| `convnextv2_tiny.fcmae_ft_in22k_in1k_384` | **CC-BY-NC-4.0** | arm C (in `export/main_v3`), arm D, all arm-N | **conflict** |
| Qwen3-32B-AWQ | Apache-2.0 | labels llm_v1 | OK |
| gemma-3-27b-it w4a16 | Gemma Terms of Use | 4th voter | outputs usable; a classifier is not a "Model Derivative" as I read the ToU; cite the ToU and Prohibited Use Policy in the release |
| pilkwang public table | hosted-API output, 3rd party | `targets/v1_1` only | do not ship |

- **ConvNeXt-V2/FCMAE is non-commercial.** Kaggle's standard winner licence is an open-source licence that does not restrict commercial use. I could not fetch the rules page (JS-locked), and `docs/research.md` does not quote the winner-licence clause, so this needs a check on the rules page or a host ruling. If it holds, any entry containing arm C, D or N is not releasable under it, and **our current main entry (main_v3) contains arm C.** Options: (a) a host ruling, (b) keep an Apache-only final (A + eff, plus a ConvNeXt-V1 tiny/small view-crop arm trained on `v1_1_gemma3`), (c) retrain arm C's recipe on `convnext_tiny.fb_in22k_ft_in1k_384`.
- **Label tables and data-use terms.** They are derived from competition reports (rule 4.b quoted in R§1). `{key}_evidence` columns and `*.jsonl` contain **verbatim report text**, so publishing them republishes competition data. Release only the state/severity columns without evidence, or only the code plus a command, after checking the data licence. `data/fewshot/*.json` contains report excerpts too. Gold labels (`is_gold`, gold-58) are competition data and must not be published.
- The release package needs: `LICENSE` (Apache-2.0 fits the code), a pinned `requirements*.txt` for the train, LLM and Kaggle envs, `docs/reproduce.md`, configs of the final arms only, `data/folds/folds_v1.csv`, target configs, LLM configs with revisions, weights plus `ensemble.json` with a model card (data, labels, metrics OOF/gold/LB, intended use, backbone licences, limitations), the generated notebook, a third-party-licences file, and the write-up and video. Kernel `is_private: true` and dataset visibility must change to public at release.

## 5. Test coverage gaps

The server collects 275 tests (50 of them for `kaggle_fetch`). Critical paths with no test:
- `scripts/export_kaggle.py`: refusing live or incomplete folds, and a silently missing fold changing arm weights.
- Consumer exception, worker hang and `get()` timeout paths. Today only the deadline-with-prior case is tested (`test_infer.py:305`).
- Cross-version compatibility: the **shipped** notebook against the **shipped** bundle. The RELEASE HAZARD test (an old kernel versus bundles with `view_crop_mm`/`head`) exists only as a manual step.
- Mixed `CacheConfig` across members; a prior-row fraction alarm.
- `assemble_oof` only on 1 synthetic fold. No test for `compare_ensemble.py`, `evaluate_runs.py`, `make_labels_v1_1.py` or `targets_v1_1_chain.sh`; the manifest `version` is not checked against the directory name.
- Resume equivalence (a resumed fold equals an uninterrupted one), which does not hold today (section 2).
- The `labels/llm.py::run_extractor` vLLM path (needs a GPU); fp16-vs-fp32 for ConvNeXt-V2 (GRN) on the T4 path.

## 6. Prioritised fix list

**P0 - could lose the competition or disqualify us**
1. Resolve the ConvNeXt-V2 FCMAE CC-BY-NC question before choosing the final entry: read the winner-licence clause and ask the host. Otherwise plan an Apache-only final and stop investing in FCMAE arms. *0.5 d plus up to about 15 GPU-h if arm C/D is retrained on ConvNeXt-V1.*
2. Make `export_kaggle.py` refuse a fold that has `last.pt`, whose `metrics.epoch` is not `epochs-1`, or whose run screen is alive. Fail rather than skip when a requested fold is missing. *2 h incl. tests.*
3. Put the tree under git on the server (the authoritative copy): commit, set `RSNA_KAD_GIT_SHA` in `run_screen.sh`, and tag every dataset/kernel push. Regenerate the local notebook or stop editing locally. *2 h.*
4. Make inference never end without a file: a per-study `get(timeout=...)` giving a prior row; a try/except per study in `_Consumer.run` giving a prior row; a top-level fallback that writes a prior-only submission. Then run a full main_v3/final kernel-vs-local end-to-end verification (the stage-10 `--verify`). *0.5 d.*

**P1 - before the final submission**
5. Assert one cache `params_sha256` across members (`infer.py:477`) and alarm on the prior-row fraction. *1 h.*
6. Add a liveness guard to `compare_ensemble.py`, `evaluate_runs.py` and `predict_oof.py`; make `--recompute` write to a new file. *2 h.*
7. Fix the `v1_1_gemma3` manifest version and script the gemma full-corpus -> merge -> targets commands under `scripts/`. *2 h.*
8. Ship a timm wheel in the dataset (or assert `timm.__version__`) and pin the kernel environment. Record an fp16/fp32 gap and the runtime for the final ensemble on 2xT4 with at most 25 members. *0.5 d.*
9. Weight the ensemble per arm in `ensemble.json`, or test that each arm has the same number of folds. *1 h.*

**P2 - release hygiene**
10. Pinned `requirements` for 3 envs, HF revisions in configs (`revision=`), `docs/reproduce.md` with the end-to-end command order, a determinism note (resume caveat, `torch.compile`, vLLM). *1 d.*
11. Release package: `LICENSE`, model card, third-party licences, the dataset licence field (not CC0 if NC-derived), label tables stripped of evidence and gold, exclusion of `qa/competitor/kernels` and `.kaggle`. *1 d.*
12. Prune superseded configs and runs into `archive/`; move production scripts out of `qa/`; parameterise the `/raid` paths in the shell scripts; drop `InferConfig`, `iter_studies` and the `ModelConfig.crop_mm` duplicate; unify `load_bundle`/`load_member`. *0.5-1 d, after the final is frozen.*
