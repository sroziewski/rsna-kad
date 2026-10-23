# RSNA Knee Abnormality Detection - architecture (2026-09-18, deadline 2026-10-22)

Inputs: `docs/research.md` (R§n) and `docs/data_analysis.md` (A§n). Everything below is consistent with
those two documents; where a number is an estimate of ours it is marked *est.* Roles downstream: the
developer implements one stage at a time, the tester verifies its acceptance tests, the reviewer audits.

## 1. Solution overview

```
train.csv (4,407 reports, 58 gold rows) ─┐
                                         ▼
  [labels/rules.py]  rule voter ──┐   [labels/llm.py] Qwen3-32B-AWQ via vLLM (GPU 0, .venv-llm)
                                  │      4-state + severity JSON per label, 2 prompt variants
                                  └──►  [labels/merge.py] ──► label table  data/labels/llm_v1.parquet
                                                                     │  (states, one row per study)
             gold-58 (validation anchor, per-fold calibration only)  ▼
                                                   [labels/targets.py] ──► data/targets/v1/fold{k}.parquet
                                                                            (soft target + weight per label)
train_series/*.dcm (819k files) ──► [cache.py] one pass, 24 procs ──► /raid/.../cache/rsna_kad/v1/
   pydicom decode, InstanceNumber sort, 130 mm crop, 384 px, ≤40 slices, per-series p1/p99 → uint8
                                                     │  index.parquet (slot, side, shapes, errors)
analysis/out/study_summary.csv ──► [cv.py] ──► data/folds/folds_v1.csv (5-fold GroupKFold on site_key)
                                                     ▼
   [data.py] study → 4 slots × K triplets ──► [model.py] shared 2-D backbone → per-slot attention-MIL
             → slot fusion with presence mask → 12 logits (+ per-slot aux heads)
   [train.py] 5 folds × {arm} ──► runs/<run_id>/fold{k}/model.safetensors + oof.parquet + metrics
                                                     ▼
   [kaggle_export.py] bundle (weights + json + wheels) ──► Kaggle Dataset  `rsna-kad-weights`
   notebooks/kaggle_inference.py (generated, self-contained) ──► Kaggle kernel: DICOM → same preprocessing
             → fp16 models on 2×T4 → submission.csv (StudyInstanceUID + 12 columns)
```

Two products from one code base: **main track** = ensemble of 5-fold × 2-3 backbone arms at 384 px;
**efficiency track** = one `tf_efficientnetv2_s` fold-average at 256 px, 3 slots, chosen by a config
switch in the same notebook script.

## 2. Decisions

### 2.1 Label extraction (reports → states)

| Topic | Decision | Rationale / rejected |
|---|---|---|
| Model | **Qwen3-32B, AWQ int4** (`Qwen/Qwen3-32B-AWQ`, ~20 GB), thinking disabled. Fallback if AWQ kernels misbehave on Ampere: `Qwen3-14B` bf16 (~29 GB). | R§2: 3-4 B models unreliable (80 % agreement, MedGemma failed JSON); R§6.1 recommends 7-32 B class; Qwen3 covers all 9 languages incl. tr/el/bg/hr. 32 B bf16 (64 GB) does not fit one A6000. Gemma-3-27B rejected (needs int4 too, weaker JSON compliance in R§2). Hosted APIs excluded (rule 4.b, R§1). |
| Serving | **vLLM in a separate venv** `/raid/sroziewski/sandbox/rsna_kad/.venv-llm` (python 3.12 `-m venv`, `pip --no-cache-dir`), offline `LLM.generate` batch, `guided_json` (xgrammar) with our schema, temperature 0, `max_tokens` 700, prefix caching on (shared system prompt + few-shots). `HF_HOME=/raid/sroziewski/cache/huggingface`. | vLLM pins torch → must not touch the shared venv (constraint). transformers + `outlines` in the shared venv rejected: 5-10× slower and no continuous batching; 4,407 × ~350 output tokens ≈ 1.5 M tokens → *est.* 30-45 min per prompt variant on one A6000 with vLLM vs > 4 h with transformers. |
| Output schema | JSON: `{"knee_side": "left|right|unknown", "postoperative": bool, "normal_template": bool, "labels": {<key>: {"state": "affirmed|hedged|negated|absent", "severity": "none|mild|moderate|severe|unspecified", "evidence": "<≤120 chars quote>"}}}` for the 12 keys in §3.2. | 4 states raised gold agreement 80.7 → 86.5 % (R§2); severity is the "central thesis" because host thresholds (R§1: ACL > 50 % fibres, meniscus surface-contacting on ≥ 2 images, effusion moderate/large only, MCL high-grade only) make "mild" mentions gold-negative. The evidence quote enables audit and a hallucination check (quote must occur in the report, fuzzy ≥ 0.8). |
| Prompt | System prompt: label definitions **with the host thresholds verbatim**, the state semantics, and the rule "mild findings below the threshold → state=affirmed, severity=mild (the mapping decides)". Few-shot: **3 same-language examples** (English fallback for fr) drawn from *unlabelled* reports, JSON written by the developer after a zero-shot draft; `assert_no_gold()` fails the run if any few-shot StudyInstanceUID is gold. Two variants: V1 = definitions-first, V2 = report-first with a short "checklist" instruction; both share the few-shots. | Same-language examples keep prompts at ~1.5 k tokens instead of ~5 k for all-language; French has no gold (A§1), so few-shot cannot depend on gold anyway. Two variants + rules give three voters; disagreement becomes uncertainty, never majority-vote consensus (three-way consensus hurt in R§2). |
| Rule voter | `labels/rules.py`: per-language term lists for the 12 labels + negation window (≤ 5 tokens, per-language negators) → state ∈ {affirmed, negated, absent}. Used (a) as MVP labels before the LLM is up, (b) to flag LLM "absent" when a term is present (`mention_conflict`). | R§2: rule extractors reach ~0.81 gold AUC - enough for a first submission and a sanity check, not for the final model. |
| Validation on gold-58 | Score per label = state→{affirmed 1.0, hedged 0.66, absent 0.33, negated 0.0} minus 0.35 if severity=mild for {menisci, effusion, MCL, OA}; macro AUC over 12 labels vs gold. Reported per language and per label; CI by bootstrap (n=58 → ±0.03, R§6.4). Acceptance ≥ 0.85, goal ≥ 0.89 (public best 0.887, R§2). | Gold rows are never in few-shots and the mapping uses no gold, so this is leak-free. |
| States → soft targets | `labels/targets.py`, per fold: base table (label-class specific) then **fold-local empirical-Bayes calibration** on the gold rows of the *training* folds (≈46): each (label, state, severity-bucket) cell gets posterior mean with Beta prior (strength 8) around the base value, then PAVA to enforce affirmed ≥ hedged ≥ absent ≥ negated. Base: affirmed+moderate/severe/unspecified 0.92; affirmed+mild: menisci/effusion/Baker's 0.25, MCL 0.20, ACL/OA 0.35, contusion/fracture 0.70; hedged 0.50; negated 0.04; absent: well-reported labels (ACL, MCL, both menisci, effusion) 0.06; under-reported labels (synovitis, fracture, contusion, Baker's, the 3 OA) = calibrated absent-rate. Voter disagreement (variants differ in state) → target = mean of the two mapped values, weight × 0.5. | JunhaoLiXD's fold-local calibration fixed base-rate collapse (R§2). Gold prevalence is enriched (A§Implications 1), so calibration only adjusts relative cell values, never global priors. |
| Per-label weights (masks) | weight ∈ [0,1] per (study, label): affirmed/negated 1.0; hedged 0.7; absent 0.5 for well-reported labels, **0.1 for under-reported labels**; `parse_ok=False` or `postoperative` → all weights 0.2; hallucinated evidence → 0.3. | Synovitis is gold-positive in 47 % but mentioned in 12 % of reports, fracture 31 % vs 5 % (R§1, A§1): "unmentioned" is not negative there. Weight 0 was rejected: a tiny weight keeps the loss defined for every label and the ranking still comes from mentioned cases. Stage 10 replaces these targets with 0.5·OOF + 0.5·prior pseudo-labels (R§3 lumbar-2nd). |

### 2.2 Preprocessing cache

| Topic | Decision | Rationale / rejected |
|---|---|---|
| Slots | 6 series types exist (A§2). **Cache all 24,371 series**; model slots (main) = `sag_fs`, `cor_fs`, `ax_fs`, `sag_t1` (FS=0); optional 5th `cor_t1`; `ax_t1` (19 %) ignored. Selection per slot (`series.py`): CSV flags only (A§2, description is 18 % dummies); among candidates prefer 2D over 3D, then most slices, then lowest SeriesNumber; **fallback** sag_fs → sag_t1 (257 studies), cor_fs → cor_t1 (159); ax_fs always present. At train time a random same-type sibling is used with p=0.3 (44 % of studies have ≥ 2). | Caching every series costs +14 % disk and buys sibling augmentation and future 5/6-slot arms without a rebuild. |
| Geometry | Sort by InstanceNumber (monotone along the normal in 100 % of series, A§3). **Constant 130 mm square crop** centred on the image centre using PixelSpacing (pad zeros when FOV < 130 mm; 640×1280 DESS handled by mm-crop), resampled to **384 px** (0.339 mm/px ≈ native 0.312) with `cv2.INTER_AREA`. | 130 mm covers 99.6 % of series (R§1); 384 is the resolution of the 0.94 family (R§2) and near-native, so nothing is lost at cache time; smaller inputs (256 px) are derived on the fly. A 90 mm meniscus crop is derived from the 384 cache by up-scaling (8 % below native resolution - acceptable, avoids a second cache). |
| Slices | 2D series: keep all if ≤ 40, else uniform subsample to 40 (p95 = 45, A§2). 3D series (837, 50-320 slices, 0.6-3.2 mm): average slabs to 3.0 mm-equivalent along z, then cap 40. | Variable N ≤ 40 keeps small-structure detail for the typical 30-slice series; a fixed 32 would drop slices from 35 % of series. Resampling 3D to 3 mm makes it look like the 2D majority. |
| Intensity | Per image apply RescaleSlope/Intercept (Philips, per-file slopes, A§4); per **series** compute p1/p99 over all voxels, clip, scale to **uint8** 0-255. Store p1/p99 in the index. | A§4: raw units differ 30×, Toshiba signed, window tags useless. uint8 halves disk vs fp16; after windowing 8 bits exceed the median 930 distinct raw values. Per-image windows rejected (z-flicker). |
| Truncated files | The 2 truncated slices (A§6): `read_slice` catches `ValueError` → slice replaced by the mean of its neighbours and `truncated_slices += 1` in the index. Same path at inference. | |
| Layout | `/raid/sroziewski/cache/rsna_kad/v1/{train,test}/<StudyInstanceUID>/<SeriesInstanceUID>.npy` (uint8, shape `(N,384,384)`, memmap-able) + `index.parquet` + `manifest.json` (params hash, counts). | One file per series keeps random-access simple and lets `np.load(mmap_mode="r")` read only sampled slices. npz/zarr rejected (no benefit at this size). |
| Disk | 24,371 series × *est.* 30 slices × 147,456 B = **~108 GB** (max 24,371 × 40 × 147 KB = 144 GB if every series hit the cap; the cap only applies to the 3D 3.4 % and the 2D tail). Budget line in §5. | Under the 150 GB cap; v2 caches must replace v1, never coexist. |
| Throughput | 24 processes, each handles one series end-to-end. Decode is a memcpy (A§4); resize dominates: *est.* 819k slices × ~4 ms / 24 ≈ 3 min CPU-bound + I/O 2 min → **< 15 min** cold. | |

### 2.3 Laterality

50 % of studies have no side tag, 25 are contradictory (A§3). `laterality.py` computes `side_tag` (majority
over series; contradictory → majority with `conf=0.6`), `side_desc` (`RT_`/`LT_`/trailing ` L`/` R` in
SeriesDescription), `side_geom` (sign of the FOV-centre x in patient coordinates from IOP/IPP of the axial
or coronal slot; the sign convention is fitted per manufacturer on tagged studies and only trusted where
agreement with tags ≥ 95 % on ≥ 30 studies - GE and Hitachi are expected to fail, R§1), and `side_est`
with `side_conf`. **MVP and main arms do not flip and do not canonicalise**: medial/lateral is anatomically
identifiable in the image (fibula, condyle shape), so the network can learn it from 4.4 k studies.
Stage 9 tests two alternatives on OOF medial/lateral AUC: (a) flip-to-left canonicalisation of coronal/axial
slots where `side_conf ≥ 0.9`, (b) horizontal flip as augmentation on coronal/axial with labels unchanged
(never sagittal - it swaps anterior/posterior). Rejected: hard-coding medial/lateral from headers (A§Impl. 5).

### 2.4 Model

| Topic | Decision | Rationale / rejected |
|---|---|---|
| Backbones | MVP: `tf_efficientnetv2_s.in21k_ft_in1k` @256. Main arms in order: A `convnext_small.fb_in22k_ft_in1k_384`, B `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, C `maxvit_tiny_tf_384.in1k`. Efficiency: `tf_efficientnetv2_s` @256. ImageNet weights via timm, downloaded once into `HF_HOME` on the server; no domain FMs. | R§5, R§6.3. Diversity comes from labels/views (arm B trains on 5 slots with the 90 mm sagittal meniscus crop as an extra view) rather than more backbones (R§2: swaps gave +0.001). |
| 2.5-D input | Per slot, K centre indices (train K=16 stratified-random, infer K=24 uniform) → triplet `[i-1, i, i+1]` as RGB (edge-clamped), uint8 → float, ImageNet mean/std after /255. | The winner recipe across RSNA 2022-2024 (R§3). |
| Pooling | Shared backbone → GAP → D (768 for convnext_small) → per-slot **gated attention MIL** (Ilse et al.; 1 shared attention over triplets, tanh·sigmoid gate, hidden 256) → slot embedding. Slot fusion: concat of 4 slot embeddings (missing slot → learned `[MISSING]` vector) + presence bits → MLP(512) → 12 logits. Aux per-slot heads (12 logits each, loss weight 0.3) for deep supervision. | Attention over slices beats max (R§3 lumbar-2nd); concat with a learned missing vector handles the 257/159 missing-FS studies explicitly. Transformer over all slots' triplets rejected (memory at 384 px, no evidence of gain). |
| Augmentation | Random 130→[115,130] mm crop scale + ±8° rotation + ±3 % brightness/contrast on the uint8 volume (same params for all slices of a series), z-jitter of the K indices, sibling-series swap p=0.3, slot dropout p=0.1 (except when it would leave < 2 slots), manifold mixup on slot embeddings p=0.3 (soft targets mix linearly). **No horizontal flip** (see §2.3). | R§2 (no flip), R§3 (manifold mixup for noisy labels). |
| Loss | BCE-with-logits on soft targets, weighted per (study, label) by the mask; reduction = weighted mean over present cells. | Ranking metric → no class balancing beyond the masks; focal rejected (calibration irrelevant for AUC). |
| Optimisation | AdamW, lr 1e-4 backbone / 1e-3 heads, wd 0.05, cosine with 1-epoch warmup, **12 epochs**, EMA 0.999/step (EMA weights are what we save), bf16 autocast on A6000, gradient checkpointing in the backbone, grad clip 2.0, `channels_last`. Batch = 4 studies × 4 slots × 16 triplets = 256 images (convnext_small @384 *est.* 30 GB; bs 3 on GPU 1 if its 38 GB shrink). | |
| Train cost | Fold train set ≈ 3,480 studies × 64 triplets = 223 k images/epoch; convnext_small fwd+bwd @384 bf16 *est.* 250 img/s → 15 min/epoch → **~3 h/fold, ~15 GPU-h per arm**; coatnet/maxvit ~1.5× that. MVP effnetv2_s @256, 3 slots, 6 epochs: ~35 min for one fold. | Two folds run concurrently on GPU 0 and GPU 1 → ~8 h wall per arm. |
| Inference precision | fp16 autocast on T4 (bf16 3× slower there, R§1). Stage 4 asserts |AUC_fp16 − AUC_fp32| < 0.003 on 200 cached studies. | |

### 2.5 Validation protocol

- **Folds**: `cv.py` → 5-fold `StratifiedGroupKFold(groups=site_key, y=lang)` on
  `analysis/out/study_summary.csv` (123 groups, largest 5.9 %, A§3), seed 2026, saved once as
  `data/folds/folds_v1.csv` and never regenerated (a test pins its md5). Gold-58 rows get folds like everyone.
- **Gold handling**: gold rows are **excluded from every image training set**. Fold k's targets are calibrated
  on gold rows of folds ≠ k; fold-k model predicts the gold rows of fold k; the 58 predictions across folds form
  the gold OOF. This is the only path by which gold touches training and it never leaks a gold row into its own
  evaluation.
- **Metrics** (`metrics.py`): (1) `oof_weak_auc`: OOF on the 4,349 weak rows, binarised per label as
  positive if target ≥ 0.6, negative if ≤ 0.2 and weight ≥ 0.5, middle band dropped; macro over 12 labels;
  (2) `oof_strict_auc`: affirmed vs negated only; (3) `gold_auc` (58, with bootstrap CI). Decisions use (1),
  (2) as a check that soft-target changes did not move both, (3) as a leak/sanity check only (R§6.4). Random
  5-fold is reported once for the MVP to quantify site memorisation (R§1: ~0.05) and then dropped.

### 2.6 Ensembling and TTA

Plain mean of sigmoid outputs across folds and arms with equal weights; no per-label or LB-tuned weights (R§6.8).
TTA (main track only): 2 views = identity + z-offset of K indices by half a stride; rotation TTA only if the
time budget in §2.7 leaves > 40 % margin. Efficiency track: no TTA, 2 folds averaged (the two best OOF folds).

### 2.7 Kaggle inference design (`notebooks/kaggle_inference.py`)

- **Self-contained by construction**: `kaggle_export.build_notebook()` inlines `constants, dicom, series,
  preprocess, laterality, model, infer` between `# --- inline:` markers; a test asserts the committed notebook
  equals the freshly built one, so preprocessing cannot drift from `cache.py`.
- **Decode path**: pydicom 3 with wheels for `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`,
  `pylibjpeg-rle`, `python-gdcm` shipped in the weights dataset (`wheels/`), installed with
  `pip install --no-index --find-links`; decoder tries pylibjpeg then gdcm; train is 100 % uncompressed (A§3)
  but the Data page warns test may be compressed (R§1). Any slice failure → neighbour fill; whole-series
  failure → slot missing; study with zero usable slots → prior prediction (train mean), never a crash.
- **Pipeline**: `multiprocessing.Pool(3)` decodes + preprocesses studies (identical code path to the cache)
  into shared-memory uint8 slot volumes, queue depth 8; two GPU consumer threads (cuda:0, cuda:1), each holds
  all models of its half of the ensemble list, fp16, `channels_last`, `torch.inference_mode`. Studies are
  ordered by descending file count so the 320-slice tail is processed first (no straggler at the end).
- **Robustness**: `/kaggle/temp` created if missing; every checkpoint load goes through `load_bundle()` which
  asserts `missing_keys == unexpected_keys == []` and compares a stored fingerprint (sum of 8 fixed weights)
  - a wrong or random init aborts before inference (R§2 pitfall 0.487). `timm.create_model(pretrained=False)`
  only. Submission is validated against `sample_submission.csv` (same rows, 12 columns, floats in [0,1],
  no NaN) before writing.
- **Time budget** (1,300 studies): decode+preprocess *est.* 1,300 × 4.5 series × 33 slices ≈ 190 k slices at
  ~8 ms on Kaggle CPU / 3 procs ≈ 9 min; GPU per model pass = 1,300 × 4 slots × 24 triplets = 125 k images at
  384 px, convnext_small fp16 on T4 *est.* 200 img/s ≈ 10.5 min; main ensemble = 15 passes (3 arms × 5 folds)
  × 2 TTA / 2 GPUs ≈ **2.6 h**; + wheels/model load 3 min → ~2.9 h; with the mandatory **25 % margin = 3.6 h**
  ≪ 9 h. Efficiency variant: 2 passes @256, 3 slots, 12 triplets → ~2 min GPU, decode ~7 min (3 series) →
  **~10-12 min total** (≈ 0.01 AUC-equivalent, R§1). Stage 4 measures the real per-study time on the 3 public
  test studies and stage 8 on a 200-study rehearsal; the ensemble list is trimmed until measured_time × 433/200
  × 1.25 < 6 h.
- **Switch**: `ensemble.json` in the dataset has `{"mode": "main"|"efficiency", "members": [...]}`; the notebook
  reads it, so the two tracks differ only by which dataset version the kernel mounts.

### 2.8 Submission mechanics

On the server (shared venv, `pip install --no-cache-dir kaggle`): `export KAGGLE_CONFIG_DIR=/raid/sroziewski/sandbox/rsna_kad/.kaggle`.
The CLI reads `access_token` / `kaggle.json` from that dir (if the installed CLI only reads `kaggle.json`, set
`KAGGLE_API_TOKEN=$(cat $KAGGLE_CONFIG_DIR/access_token)` - verify in stage 4 [?]). Weights:
`kaggle datasets create -p export/<bundle>` (first time, `dataset-metadata.json` id `<user>/rsna-kad-weights`),
then `kaggle datasets version -p export/<bundle> -m "<run_id>"`. Kernel: `notebooks/kernel/kernel-metadata.json`
(`code_file: ../kaggle_inference.py`, `enable_gpu: true`, `enable_internet: false`, `dataset_sources`,
`competition_sources`), `kaggle kernels push -p notebooks/kernel`, `kaggle kernels status`, `kaggle kernels output`.
Selecting a kernel version as a competition submission is a manual click in the Kaggle UI (code competition).
`scripts/kaggle_cli.sh` wraps all of this with the env var set and never writes under `$HOME`.

## 3. Repository layout and data contracts

### 3.1 Layout

```
src/rsna_kad/
  constants.py     LABELS (12 CSV names, order of sample_submission), LABEL_KEYS, SLOTS, DATA_DIR, CACHE_DIR
  config.py        dataclasses CacheConfig, TargetsConfig, ModelConfig, TrainConfig, InferConfig;
                   load_config(path: str) -> TrainConfig   (YAML → dataclasses, unknown keys raise)
  dicom.py         read_header(path) -> Header; read_slice(path) -> np.ndarray | None (truncated → None);
                   sort_series(paths) -> list[str]; series_geometry(headers) -> Geometry (spacing, iop, ipp_centre, plane, is_3d, z_step)
  series.py        slot_of(plane, fs) -> str; assign_slots(series_df) -> DataFrame (study, slot, series_uid, rank, fallback);
                   select_for_slot(candidates) -> str
  preprocess.py    crop_resample(img, spacing_mm, crop_mm, px) -> float32; resample_z(vol, z_step, target_mm) -> vol;
                   robust_window(vol, p_lo, p_hi) -> (uint8, lo, hi); series_to_volume(slices, geom, cfg) -> (uint8[N,px,px], meta)
  laterality.py    estimate_side(headers_by_series, series_df) -> Side(side_tag, side_desc, side_geom, side_est, side_conf)
  cache.py         build_cache(data_dir, out_dir, split, cfg, workers) -> index_df; load_series(cache_dir, split, study, series) -> memmap;
                   load_study(cache_dir, split, study, slots, sibling_rng=None) -> dict[slot, memmap]; verify_cache(cache_dir) -> report
  cv.py            make_folds(study_summary_csv, n_splits, seed) -> folds_df; load_folds(path) -> folds_df
  labels/schema.py ReportLabels dataclass + JSON_SCHEMA (for guided decoding) + parse_output(text) -> ReportLabels | None
  labels/rules.py  extract_rules(report, lang) -> dict[key, state]; TERMS[lang][key], NEGATORS[lang]
  labels/prompts.py build_messages(report, lang, variant) -> list[dict]; load_fewshot(lang) -> list; assert_no_gold(fewshot_ids, gold_ids)
  labels/llm.py    run_extractor(reports_csv, cfg, variant, out_path) (vLLM, runs in .venv-llm); resume from partial output
  labels/merge.py  merge_votes(llm_v1, llm_v2, rules) -> label_table (agreement flags, mention_conflict, hallucinated_evidence)
  labels/targets.py state_prior_table(cfg) -> DataFrame; calibrate(label_table, gold_df, train_ids, cfg) -> Calibration;
                   make_targets(label_table, calib) -> targets_df; gold_state_auc(label_table, gold_df) -> DataFrame
  data.py          StudyDataset(cache_dir, targets_df, folds_df, cfg, train: bool); collate_studies(batch) -> dict;
                   augment_volume(vol_u8, rng, cfg) -> vol_u8
  model.py         KneeMIL(nn.Module); build_model(cfg) -> KneeMIL; save_bundle(model, meta, dir); load_bundle(dir, device) -> (model, meta)
  train.py         train_fold(cfg, fold, run_dir) -> metrics; predict_studies_cached(model, dataset, tta) -> DataFrame
  metrics.py       macro_auc(y_soft, y_pred, weight, pos_thr, neg_thr) -> (macro, per_label); gold_report(pred58, gold) -> dict
  infer.py         StudyPreprocessor(cfg) (raw DICOM dir → slot volumes); predict_studies(members, study_dirs, devices, workers) -> DataFrame;
                   write_submission(pred_df, sample_csv, out_csv)
  kaggle_export.py export_bundle(run_dirs, out_dir, mode) ; build_notebook(out_path) ; push_dataset(dir) ; push_kernel(dir)
configs/           cache_v1.yaml, targets_v1.yaml, folds_v1.yaml, llm_qwen3_32b_awq.yaml, mvp_effv2s_256.yaml,
                   main_convnext_s_384.yaml, main_coatnet2_384.yaml, main_maxvit_t_384.yaml, eff_effv2s_256.yaml
scripts/           build_cache.py, make_folds.py, extract_labels_rules.py, extract_labels_llm.py (.venv-llm), make_targets.py,
                   train.py (--config --fold --gpu), predict_oof.py, export_kaggle.py, bench_infer.py, run_screen.sh, kaggle_cli.sh
notebooks/         kaggle_inference.py (generated), kernel/kernel-metadata.json
tests/             test_dicom.py, test_series.py, test_preprocess.py, test_cache.py, test_laterality.py, test_labels_rules.py,
                   test_labels_schema.py, test_targets.py, test_cv.py, test_data.py, test_model.py, test_metrics.py, test_infer.py,
                   test_export.py, conftest.py (synthetic DICOM series written with pydicom: 2D, 3D, non-square, truncated)
data/              folds/folds_v1.csv, labels/*.parquet, targets/v1/fold{k}.parquet, fewshot/{lang}.json  (produced on the server,
                   small files pulled back into the repo with a new `remote.ps1 pull` action; `sync` must add `data`)
runs/, export/, logs/   server-only (never synced)
```

Conventions: YAML configs + dataclasses (chosen over pure-python configs so a run's config can be copied into the
bundle verbatim). Long jobs: `scripts/run_screen.sh <name> <cmd>` → `screen -dmS <name>` with
`logs/<name>_<timestamp>.log`, `HF_HOME=/raid/sroziewski/cache/huggingface`, `PIP_NO_CACHE_DIR=1`,
`PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES` explicit. Tests: `python -m pytest tests -q -p no:cacheprovider`
with `TMPDIR` inside the project (as `remote.ps1 test` already does). Every script is idempotent/resumable.

### 3.2 Data contracts

**Label keys** (order fixed, maps 1:1 to the CSV names): `acl, mcl, med_men, lat_men, med_oa, lat_oa, pf_oa,
effusion, synovitis, baker, contusion, fracture` ↔ `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.

**Label table** `data/labels/<extractor>_<version>.parquet` (+ `.csv` copy), exactly 4,407 rows, one per study:
`StudyInstanceUID: str` (unique) · `lang: str` · `extractor: str` · `variant: str` · `parse_ok: bool` ·
`knee_side: category{left,right,unknown}` · `postoperative: bool` · `normal_template: bool` · for each key
`{key}_state: category{affirmed,hedged,negated,absent}` · `{key}_severity: category{none,mild,moderate,severe,unspecified}`
· `{key}_evidence: str`. Merged table adds `{key}_agree: bool`, `{key}_mention_conflict: bool`,
`{key}_evidence_ok: bool`. Gold rows are present (the extractor runs on all reports) and flagged by joining `folds`.

**Targets** `data/targets/<version>/fold{k}.parquet`, 4,349 rows (gold excluded): `StudyInstanceUID`,
`{key}_target: float32 ∈ [0,1]`, `{key}_weight: float32 ∈ [0,1]` × 12; sidecar `calibration_fold{k}.json`.

**Folds** `data/folds/folds_v1.csv`: `StudyInstanceUID, fold:int(0-4), site_key:str, lang:str, is_gold:bool`; 4,407 rows.

**Cache**: `<cache>/v1/<split>/<StudyInstanceUID>/<SeriesInstanceUID>.npy`, dtype uint8, shape `(N, 384, 384)`,
1 ≤ N ≤ 40. `index.parquet` one row per series: `split, StudyInstanceUID, SeriesInstanceUID, plane, fs:int, slot,
rank:int, selected:bool, is_3d:bool, n_src, n_cached, spacing_mm, fov_mm, z_step_mm, crop_mm, px, p_lo, p_hi,
truncated_slices:int, side_tag, side_desc, side_geom, side_est, side_conf:float32, ok:bool, error:str`.
`manifest.json`: `{version, params, params_sha256, n_series, n_ok, built_at}`.

**OOF** `runs/<run_id>/oof.parquet`: `StudyInstanceUID, fold, {key}_pred: float32 × 12` (4,407 rows: weak + gold);
`runs/<run_id>/metrics.json`: `{oof_weak_auc, oof_strict_auc, gold_auc, per_label: {...}, gold_ci95}`.

**Checkpoint bundle** `runs/<run_id>/fold{k}/`: `model.safetensors` (EMA weights) + `bundle.json`
`{run_id, fold, arch: {backbone, img_px, crop_mm, slots, K_infer, attn_hidden, fusion}, norm: {mean, std, p_lo, p_hi},
labels: [12 CSV names], fingerprint: [8 floats], config: <TrainConfig as dict>, metrics: {...}, git_sha}`.
Export dir = Kaggle dataset root: `models/<arm>_f<k>.safetensors`, `models/<arm>_f<k>.json`, `ensemble.json`
`{"mode", "members": [{"path", "weight", "tta"}], "K_infer", "workers"}`, `wheels/*.whl`, `VERSION`.

**submission.csv**: header `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,
Effusion,Synovitis,Baker's,Contusion,Fracture`; one row per test study in `sample_submission.csv` order; float32
probabilities in [0,1]; no NaN.

## 4. Work plan (stages, executed one at a time)

| # | Stage / deliverables | Acceptance tests (automatable) | Est. wall |
|---|---|---|---|
| 0 | **Scaffold**: `src/rsna_kad/{constants,config}.py`, `pyproject.toml` (editable install into the shared venv), `configs/cache_v1.yaml`, `tests/conftest.py` synthetic DICOM fixtures, `scripts/run_screen.sh`, `remote.ps1` gains `pull` + `data` in `sync`. | `remote.ps1 test` green on the server; `python -c "import rsna_kad"` works from the shared venv; a synthetic study with 2D, 3D (0.9 mm × 120 slices), non-square (640×1280) and one truncated file is generated in < 5 s. | 0.5 d |
| 1 | **Cache + folds**: `dicom.py, series.py, preprocess.py, laterality.py, cache.py, cv.py`, `scripts/build_cache.py, make_folds.py`, `data/folds/folds_v1.csv`. | Unit: truncated file → None and neighbour fill; 3D series resampled to ≤ 40 slices at 3 mm; crop of a 205 mm FOV and a 120 mm FOV both give (N,384,384) with correct mm/px; InstanceNumber order equals IPP order. Integration on the server: **all 24,371 train series + 15 test series cached in < 60 min on 24 procs**, `index.parquet` has 24,386 rows, `ok` for all but 0 (the 2 truncated files report `truncated_slices=1`, `ok=True`), total size 90-145 GB, every `.npy` has shape `(N,384,384)` with 1 ≤ N ≤ 40, per-series mean intensity in [20, 200] for ≥ 99 % of series. Folds: 5 folds, each 18-22 % of studies, no `site_key` in two folds, each fold holds 8-16 gold rows, md5 pinned in `tests/test_cv.py`. `side_geom` agrees with `side_tag` on ≥ 95 % of tagged Siemens/Philips studies. | 2 d |
| 2 | **Rule labels + target pipeline v0**: `labels/{schema,rules,targets}.py`, `metrics.py`, `scripts/extract_labels_rules.py, make_targets.py`, `data/labels/rules_v0.parquet`, `data/targets/v0/`. | `gold_state_auc(rules_v0)` macro ≥ 0.75 on gold-58; table has 4,407 rows and the schema of §3.2; targets have 4,349 rows, all values in [0,1], zero NaN; the Turkish normal template (37 rows) gets negated states on ≥ 8 labels; `macro_auc` unit-tested against sklearn on a toy case incl. dropped middle band. | 1 d |
| 3 | **Model + MVP training**: `data.py, model.py, train.py`, `configs/mvp_effv2s_256.yaml`, `scripts/train.py, predict_oof.py`. | `KneeMIL` forward on a 2-study batch with one missing slot returns (2,12) + aux (2,4,12) and is finite; **1-fold smoke on 200 studies runs < 10 min on one A6000 and OOF-weak AUC on those > 0.55**; full MVP fold 0 (effnetv2_s @256, 3 slots, 6 epochs) finishes < 1 h; OOF-weak macro AUC ≥ 0.75 and gold-58 (fold-0 gold rows) reported; bundle saved with `bundle.json` and reloads bit-exact. | 2 d |
| 4 | **MVP submission**: `infer.py, kaggle_export.py`, `notebooks/kaggle_inference.py`, `notebooks/kernel/`, `scripts/export_kaggle.py, bench_infer.py, kaggle_cli.sh`. | Preprocessing parity: `StudyPreprocessor` output on 20 cached train studies is identical (uint8 equality ≥ 99.9 % of voxels) to the cache; **notebook run locally on the 3 public test studies produces a valid submission.csv in < 60 s per study** (fp32 CPU fallback also passes); fp16-vs-fp32 AUC gap < 0.003 on 200 studies; build_notebook reproduces the committed file byte-for-byte; a corrupted `.safetensors` (one tensor removed) aborts with the fingerprint assert; wheels install with `--no-index` in a fresh venv; dataset uploaded, kernel pushed with internet off, **public LB score exists** (expect ~0.80-0.85). | 1.5 d |
| 5 | **LLM extractor**: `.venv-llm` with vLLM, `labels/{prompts,llm,merge}.py`, `data/fewshot/*.json`, `configs/llm_qwen3_32b_awq.yaml`, `scripts/extract_labels_llm.py`, `data/labels/llm_v1.parquet`. | `assert_no_gold` unit test; guided decoding parses for ≥ 99.5 % of 4,407 reports per variant; evidence quote found in the report for ≥ 95 % of affirmed/negated cells; full pass per variant < 2 h on GPU 0; **gold-58 macro AUC ≥ 0.85 (goal ≥ 0.89)** for the merged table with gold rows excluded from few-shots; per-language AUC printed for en/es/tr; per-label affirmed rate for synovitis and fracture reported against gold prevalence. | 3 d |
| 6 | **Calibrated targets + main arm A**: `targets_v1` with fold-local calibration, `configs/main_convnext_s_384.yaml`, 5 folds of arm A. | Calibration JSON respects the monotone order for all 12 labels; targets v1 differ from v0 in ≥ 30 % of cells; **5-fold OOF-weak macro AUC ≥ 0.85 and gold-58 ≥ 0.85**; medial-side labels ≥ 0.85 each on OOF-weak; per-fold train time logged; second MVP-style submission from the 5 folds scores above the MVP. | 4 d (15 GPU-h) |
| 7 | **Efficiency variant**: `configs/eff_effv2s_256.yaml`, 5 folds, `ensemble.json mode=efficiency`. | OOF-weak ≥ 0.83; `bench_infer.py` on 200 cached-then-re-decoded studies extrapolated to 1,300 < 15 min on the T4 assumption (measured A6000 time × 4.5); public efficiency submission made. | 1.5 d (3 GPU-h) |
| 8 | **Arms B, C + ensemble + hardened inference**: coatnet/maxvit configs, arm B with 5 slots + 90 mm sagittal crop view, ensemble OOF, rehearsal. | Each arm's OOF-weak ≥ arm A − 0.01; ensemble OOF-weak ≥ best single + 0.005; 200-study rehearsal with synthetic JPEG-2000 re-encoded copies of 20 studies decodes correctly; **projected main-track time × 1.25 < 6 h**, members trimmed otherwise; main submission on LB. | 5 d (35 GPU-h) |
| 9 | **Laterality + noise experiments**: flip-canonicalisation vs flip-aug on arm A, confident-learning relabel of the 2 % most contradicted OOF cells, pseudo-labels 0.5·OOF+0.5·prior for under-reported labels, second-round training. | Each experiment is one config diff and one OOF report; kept only if OOF-weak macro +0.003 and gold does not drop > 0.02. | 5 d (30 GPU-h) |
| 10 | **Freeze + reproducibility**: final `ensemble.json` for both tracks, `docs/reproduce.md` (commands per stage), all tests green, dataset versions and kernel versions recorded. | `remote.ps1 test` green; `export_kaggle.py --verify` re-runs the notebook on the 3 public studies from the exported bundle and matches the last OOF-derived predictions to 1e-4; final submissions selected in the UI by 2026-10-20. | 2 d |

Stages 3-4 give the MVP by ~day 7 (2026-09-25); stage 5 can run on GPU 0 in parallel with stages 3-4 (which use
GPU 1); stages 6-9 hold ~28 days of calendar with ~85 GPU-h of training, leaving 2 days of slack before 10-22.

### 4.1 Stage log

- **Stage 0-1 ACCEPTED 2026-09-18** (`qa/stage1/report.md`): cache v1 = 24,386 series, 98.2 GiB, built in 265 s;
  folds_v1 md5 `4dc54b74429f07051af2175e7bb35442`. Deviation accepted: folds come from a seeded greedy balancer
  (`cv.make_folds(method="balanced")`), because sklearn `StratifiedGroupKFold` failed the size/gold criteria for all
  200 seeds tried (language ≈ site). Follow-ups owned by later stages: (F1/F5, stage 4) `process_series` must catch
  `slot_of` and pydicom `RuntimeError` for undecodable syntaxes so one bad series never aborts a run; (F3, cache v2)
  p1/p99 must exclude the zero padding of the 106 series with FOV < 130 mm; (F4, before any 6-slot arm) blacklist
  the 13-19 Philips 300 mm `ax_t1` series whose centre crop is mostly air; (F7, stage 9) GE side geometry is
  consistent per site with opposite signs - a per-site fit could recover most of the 664 unknown-side GE studies.

- **Stage 3 ACCEPTED 2026-09-18** (`qa/stage3/report.md`): MVP `runs/mvp_effv2s_256` 5 folds, OOF-weak 0.846
  (sklearn recomputation 0.845), gold-58 0.847 [0.815, 0.877], 40 min/fold, 2.7 GB peak, 470 img/s. Leakage audit
  clean. Follow-ups: (F1, stage 6) best-epoch selection on the validation fold adds ~+0.008 optimism - main arms
  report last-epoch (or fixed-epoch) OOF for decisions; (F2, metrics) `binarise` compares float32 targets against
  float64 thresholds, so targets exactly at NEG_THR (mcl affirmed-mild = 0.20) fall into the dropped band - make the
  boundary explicit; EMA decay ramp never reaches 0.999 within a 6-epoch run (documented, acceptable).

- **Stage 2+5 delivered 2026-09-18** (tester pending): rules_v0 gold 0.816; Qwen3-32B-AWQ V1 0.850 / V2 0.838 /
  merged `data/labels/llm_v1` **0.852 [0.818, 0.884]** (synovitis 0.67, contusion 0.75 weakest; tr 0.74 on n=6);
  parse 100 % / 99.93 %, evidence 97 %; V1 98 min, V2 269 min (report-first layout defeats prefix caching - relayout
  before any rerun). Targets v1 in `data/targets/v1/` (fold-local EB calibration, monotone). Under-reporting
  reproduced: synovitis affirmed 12.5 % vs 46.6 % gold, fracture 5.1 % vs 31 %.
- **Stage 2+5 ACCEPTED 2026-09-18** (`qa/stage5/report.md`): contracts and gold AUC reproduced exactly; spot-check
  of 25 reports: 8 % of cells wrong (absent<->negated on group statements, "degeneration" negated, wrong-structure
  attribution). Incident: the vLLM run wrote JIT caches and telemetry state under `~` (removed; `run_screen.sh` now
  sets `HOME`/XDG to `<proj>/.home` and `VLLM_NO_USAGE_STATS=1` - **every LLM rerun must go through
  `run_screen.sh`**). Follow-ups for targets v1.1 (before stage 8): BUG-1 a parse-failed V2 counts as disagreement
  (3 rows); BUG-2 torn last jsonl line re-runs one study on resume (harmless); prompt fixes for the two systematic
  error patterns; V2 relayout for prefix caching.

- **Stage 4 delivered 2026-09-19**: kernel `sroziews/rsna-kad-mvp` v4 on dataset `sroziews/rsna-kad-weights` v1
  (5 MVP folds) -> **public LB 0.865** (submission 56345216; hidden rerun 25 min on 2xT4; 0.53 s/study on an A6000).
  Parity 100 % bit-exact, fp16/fp32 gap 5e-5, 211 tests green. Lessons: Kaggle mounts inputs under
  `/kaggle/input/{datasets,competitions}/...`; the driver must not keep all studies' triplets (v2 OOM'd on the hidden
  set); the kaggle CLI reads only `KAGGLE_API_TOKEN` / `~/.kaggle/access_token` (`kaggle_cli.sh` handles it);
  Kaggle image = Python 3.12, torch 2.10. Submissions go through `scripts/kaggle_cli.sh submit` (no UI step).

- **Stage 6 arm A trained 2026-09-19** (`runs/main_convnext_s_384`, tester pending): throughput fixed 74 -> 200 img/s
  (12 GB peak, 3 h/fold, 15 h chain). 5-fold **OOF-weak 0.921, strict 0.852, gold-58 0.890 [0.858, 0.920]**
  (last-epoch policy; best-epoch would read ~+0.008). Weakest labels: lat_oa 0.863, mcl 0.869. Exported as
  `export/armA_v1` -> dataset v2, kernel v7, submission 56424502 = **public LB 0.931** (MVP was 0.865; public
  notebook cluster 0.936-0.941, top 0.957). Lesson: a kernel pushed seconds after
  a dataset version mounts the previous version - wait for `dataset-status` = ready, and verify the kernel output
  against local predictions before submitting.

- **Stage 7 delivered 2026-09-21** (`runs/eff_effv2s_256`, effnetv2_s@256, 3 slots, targets v1, 8 epochs, last):
  OOF-weak **0.9080**, strict 0.8388, gold **0.8666 [0.833, 0.898]**; 54 min/fold, 2.7 GB, 470 img/s. Costs only
  -0.0135 weak vs arm A at 1/7 the training and 1/3 the inference cost, and beats arm A on synovitis (0.947 vs
  0.908). **Labels, not resolution, were the dominant lever**: the same 256 px/3-slot recipe went 0.8815 -> 0.9080
  OOF-weak purely by swapping rule targets (v0) for LLM targets (v1). Open: efficiency-track submission.
  **[Corrected 2026-09-26, reviewer]** "purely by swapping targets" is false as worded: the same config change also
  went 6 -> 8 epochs and best-epoch -> last-epoch selection. The direction (labels mattered a lot) stands; the
  size attributable to labels alone was never isolated.
- **Stage 8 arm B ABORTED 2026-09-21** (`runs/main_coatnet2_384/fold0`, stopped at epoch 2): coatnet_rmlp_2_rw_384
  ran at 55 img/s (3.7x slower than arm A, not the 1.5x assumed) and *underfit* - train loss 0.685 vs arm A's 0.602
  at epoch 2. Ranked causes: flat lr 1e-4 with no layer-wise decay on a 73 M hybrid backbone (public recipes use
  1e-5..5e-5 + LLRD 0.7-0.8); 2.6x less training signal after the forced K/epoch cuts; drop_path 0.2 too high for a
  short schedule; BatchNorm running stats double-updated by gradient checkpointing (convnext is LayerNorm-only, so
  arm A is immune). Also measured: CoAtNet's stage-0 MbConv hits cuDNN's 32-bit index cap above 112 images per
  forward, so **any coatnet arm is capped at 1 study per forward at k_infer 24** - an inference-time constraint too.
  Relaunch only after a 200-study/2-epoch sweep beats the recorded curve.

- **LLM precision/prompt A-B, 2026-09-21** (`qa/llm_ab/report.md`, 481-study fixed subset): quantisation is NOT the
  bottleneck - AWQ int4 gold 0.850 vs bf16 TP=2 0.847 vs W8A16 0.845; paired sign test on disagreeing gold cells
  13-17, p=0.58. bf16 would cost 2.3x the GPU time for -0.003. **Keep AWQ int4.** More important: the "strict"
  prompt that fixed the stage-5 audit's absent<->negated pattern made gold agreement significantly *worse*
  (p=0.03/0.02) - of the cells it moved negated->absent, 93.8 % are gold-negative (OA: 100 %), because the host
  graded borderline findings negative, so a confident `negated` (target 0.04, w 1.0) beats `absent` (calibrated,
  w 0.1 for under-reported keys). **The stage-5 audit's error pattern is an error against the report text but a
  regression against the image-derived gold - do not "fix" it.** The degeneration rule is ignored at every
  precision. Carry-forward for targets v1.1 only: port the quoting and attribution paragraphs of the `strict`
  prompt (quote_miss -36 %, misattribution -18 %), never STEP 1/STEP 2. `prompts.py` now has a `rules` selector
  (default byte-identical to stage 5, md5-pinned by a test); `llm.py` supports tensor_parallel_size/cpu_offload_gb.

- **Stage 7/8 submissions 2026-09-21** (`qa/ensemble/report.md`): blend analysis on 2,000 paired bootstraps -
  mean(armA+eff) OOF-weak **0.9292** (+0.0077 over arm A, CI [+0.0051,+0.0103], P=1.000), strict 0.8614, gold
  0.8964; beats both singles on 11/12 labels (only contusion regresses, not to be "fixed" with a per-label weight
  per §2.6). MVP excluded (+0.0012, P=0.862 - not demonstrable). Rank-average loses to mean. Per-label prediction
  correlation 0.832 (mcl 0.693 = most diverse). Submissions: **efficiency `export/eff_v1` = LB 0.913**
  (K_infer 16 - measured: K12 costs -0.0033, outside the 0.002 gate; 0.423 s/study on A6000 -> ~41 min under the
  conservative T4 x4.5 rule, ~20 min by the empirical x2.2 factor from stage 4); main `export/main_v2` (10 members)
  = **LB 0.929**, ~1.6 h on 2xT4. Verification lesson extended: poll the dataset FILE LISTING, not `dataset-status`
  (v4 reported ready while still listing v3's VERSION).
- **OOF-vs-LB calibration, 2026-09-21 - READ BEFORE TRUSTING ANY SMALL DELTA.** Four scored submissions:
  MVP OOF-weak 0.8815 -> LB 0.865; eff 0.9080 -> 0.913; arm A 0.9215 -> **0.931**; ensemble 0.9292 -> 0.929.
  The blend improved OOF (+0.0077, P=1.000), strict (+0.009) and gold (+0.006) yet did **not** improve the LB
  (-0.002). LB - OOF across the three v1-trained entries: +0.005, +0.0095, -0.0002. With ~390 public studies the
  LB's own noise is ~+-0.01, so it cannot resolve anything below ~0.02 - exactly R§6.8's warning. **Consequences:**
  (1) rank candidates on OOF-weak + strict + gold together, never on the public LB; (2) do not spend submissions on
  A/B comparisons that OOF says are worth < 0.02; (3) the ensemble stays the main entry because three independent
  offline metrics agree and the private LB (~900 studies) decides. The ensemble kernel was verified correct
  (10 members loaded, kernel vs local diff 2.1e-4), so this is measurement noise, not a broken submission.
- **Latent inference bug, fix before exporting any view-crop arm**: `infer.py` does not implement
  `arch.view_crop_mm`; `model.load_bundle` refuses such a bundle unless `views=True` but `infer.load_member` has no
  equivalent guard, so a view-crop member would silently receive uncropped duplicate views at inference.
- **Stage 8 arm C 2026-09-21** (`runs/main_armc_convnextv2_t_384`, fold 0 done, chain lands ~05:25 2026-09-22):
  view-diverse arm = `convnextv2_tiny.fcmae_ft_in22k_in1k_384` over **5 views/4 series** (MAIN_SLOTS + a second
  `sag_fs` cropped to 90 mm, i.e. the centre 69.2 % of the 130 mm cache resized back to 384), 10 epochs.
  Fold 0: weak **0.9282** / strict 0.8558 / gold 0.8985 vs arm A's 0.9207 / 0.8527 / 0.8996, at **2.4 h/fold vs
  3.0** and 302 img/s vs 200; leads arm A at every matched epoch. Paired (fold 0, 500 bootstraps): A+C - A
  **+0.0137 weak [+0.0071,+0.0207] P=1.000**; A+E+C - A+E +0.0055 weak / +0.0078 strict, both P=1.000; best combo
  **A+E+C 0.9370 weak / 0.8673 strict / 0.9144 gold**. Arm C repairs arm A's synovitis (0.944 vs 0.838), loses a
  little on lat_oa and baker. **Diversity note: rho(A,C)=0.81 > rho(A,E)=0.73 yet C contributes more - prediction
  correlation alone does not rank arms.** The 5th slot `cor_t1` was swept and rejected (4x the convergence penalty
  of the 90 mm view, missing in 22.7 %, duplicates cor_fs in 159 studies). Smoke AUC at 200 studies/2 epochs has a
  +-0.04 noise floor, so the sweep was decided on train loss, throughput and memory instead - record that method.
  Inference-side view crop implemented with a byte-exact training/inference parity test; suite 234 green.
- **RELEASE HAZARD from 2026-09-21 on**: bundles now carry `view_crop_mm` in `arch`/`config.model` and
  `config.from_dict` rejects unknown keys, so a newly exported bundle **crashes the previously published kernel**.
  Dataset and kernel must be versioned together in one release step from now on, and the final submission must be
  re-verified end to end (kernel output vs a local run of the same bundle) before the 2026-10-22 deadline.
- **Stage 9 design, 2026-09-21** (`qa/stage9/report.md`): (a) **Fold-honest pseudo-labels from the existing 5 OOF
  models are impossible, not merely expensive**: a pseudo-label for study s used by the fold-k model needs a model
  excluding both fold(s) and fold k, and each of M_0..M_4 excludes exactly one fold, so the usable (s,k) set is
  empty. Honest 5-fold pseudo-labels need ten leave-2-out sources (~20 GPU-h); the experiment therefore runs on one
  held-out fold with two half-data sources, with the split proved mechanically via `train.split_studies`.
  (b) **Confident-learning denoising REJECTED on evidence**: only 52 of 52,188 cells (0.10 %) are strong
  contradictions at 0.9/0.1, none for mcl/med_oa/lat_oa/synovitis/contusion, and of 10 hand-checked cells only 1 is
  a real LLM error while 5 are correct labels the model gets wrong - down-weighting would delete correct
  supervision on the hardest cases. Saved ~10 GPU-h. (c) Pseudo-label headroom is concentrated in synovitis
  (3,700 silent cells, target 0.29 vs OOF 0.37) and pf_oa (985, 0.11 vs 0.34); the other five under-reported keys
  barely move. (d) 4 of the 10 hand checks are labels correct against the report but below the host's severity
  threshold (partial ACL tears, small-extent PF defects) -> a **targets v1.1 calibration** follow-up.
- **Tooling fix 2026-09-21**: `remote.ps1 sync` no longer pushes `data/` (it is produced on the server and pulled
  back); `-PushData` opts in. Pushing it would have overwritten label tables and targets in use by running chains.

- **targets v1.1 - fourth voter, 2026-09-21** (`qa/competitor/public_table.md`, `qa/llm_ab/voter_gain_public.md`,
  `data/targets/v1_1/`). **PATH 2 reproduced and cleared.** `pilkwang/rsna-knee-llm-labels` covers **4,406 of
  4,407** studies (one missing, and **57 of 58 gold rows** - every gold-58 number against it is really gold-57).
  Its schema is a 5-level ordinal, not a score: `conf` is a function of `verdict` (NO 0.08/0.85, UNK 0.28/0.05,
  YES 0.68|0.82|0.94 /0.95) and carries no extra information. Mapped onto §2.1: NO -> negated, UNK -> absent,
  YES 0.94 -> affirmed/severe, 0.82 -> affirmed/unspecified, **0.68 -> hedged** (measured: `hedged` 0.8567 vs
  `mild` 0.8426 alone, 0.8659 vs 0.8581 merged - their weakest YES really is a hedge). Raw-score gold-57
  **0.870** vs our `llm_v1` **0.8515**; rank-average **0.8756**, paired **+0.0243 [+0.0139,+0.0351] P=1.000**
  (the recorded +0.0270 was computed on a different ranking population; direction, size and P reproduce).
  The state-mapped 4-voter merge - the estimator that actually feeds `make_targets` - scores **0.8662,
  +0.0149 [+0.0052,+0.0248] P=0.998**, so ~0.013 of the rank-average gain is lost translating their 3-level YES
  into our 2 cells. Extractor correlation 0.892 (fracture 0.716, contusion 0.769).
  **No gold leak** (research.md §2 pitfall, 5 tests): verdict accuracy on decided gold cells is only **0.771**
  (a verbatim copy would be ~1.0); their agreement with our llm_v1 is *higher* on gold rows than off them
  (0.912 vs 0.889 - a copied block would agree less); YES cells with no rule-voter mention are equally rare on
  and off gold (0.045 vs 0.047). The one flag, a gold-row positive-rate gap of 0.157, **is enrichment, not
  copying: our own llm_v1 shows 0.159 on the same rows** - always run that control before calling a leak. Their
  raw-score ACL AUC of 0.997 is a granularity artefact (0.941 once mapped to states), not evidence of a leak.
  Caveat that survives all of this: their labels came from a hosted Anthropic API, so `analysis.md` §6's
  **host ruling is still required** before the public table can carry the final entry - which is why PATH 1 runs.
  **Machinery**: voters are now n-ary. `merge.add_voter/voter_suffixes/voter_agreement`, `labels/public.py`,
  and `make_targets` averages every voter's mapped value and grades the weight by the majority share
  `m = d + (1-d)(2f-1)`, which is *exactly* v1's rule at two voters - `data/targets/v1` regenerates bit-identical
  under the new code (verified per fold, targets and calibration). `_v3` is invisible to `metrics.voter_codes`,
  so `consensus`/`strict` stay the same yardstick across target versions (pinned by a test).
  **targets v1.1** = llm_v1 (V1+V2) + rules + pilkwang `_v3`, everything else identical to v1 (fold-local EB
  calibration on training-fold gold, PAVA, the same weights and under-reported handling; `configs/targets_v1_1.yaml`
  differs from v1 only in `version`). 27.4 % of cells move, mean |dtarget| 0.017, weight mass 0.695 -> 0.648,
  weak subset 30,730 -> 29,690 cells. Contract checked: 5 x 4,349 rows, float32 in [0,1], gold excluded,
  monotone calibration, sidecars. `data/targets/v1` and `data/labels/llm_v1*` untouched.
  **Prompt carry-forward**: `prompts.PROMPT_RULES` gains `base_qa` = `base` byte-identical **plus** the `strict`
  attribution and quoting paragraphs only. Two clauses of the ported text were dropped because they are
  themselves state rules ("if no sentence names this structure the state is absent", "if you cannot copy such a
  span the state is absent") - that is the exact mechanism the A/B measured as a regression. Used by PATH 1;
  it cannot apply to targets v1.1, whose LLM columns were extracted with `base`.
  **PATH 1 queued, not yet measured**: GPU 0 was held by `armA8` -> `stage9` for the whole session, so
  `scripts/targets_v1_1_chain.sh` (screen `tv11`) polls and then runs gemma-3-27b-it W4A16 -> Mistral-Small-3.1-24B
  W4A16 on the 481-study subset, then arm C fold 0 on targets v1.1, then a full-corpus pass **only** if a
  candidate's 4-voter merge clears >= 0.010 with P >= 0.95. Suite 258 green.

### 4.2 Decision rule (from `qa/arbiter/report.md`, 2026-09-21) - USE THIS FOR EVERY REMAINING CHOICE

Primary arbiter stays **`oof_weak_auc`**, with **`consensus`** as a mandatory companion and as the *primary* whenever
two runs used different target versions (weak-AUC is defined by the target table, so it moves its own yardstick when
targets v1.1 / stage-9 land). **`gold58` is a veto, not a score.** Accept a change at **>= 0.010 with P >= 0.95**;
at 0.005-0.010 only if weak+strict+consensus agree in sign and it costs nothing; **below 0.005 it is a tie whatever
P says**. Paired sampling noise is +-0.0026..0.0066 (gold-58 +-0.027), and the same delta moves up to 0.012 when the
label definition changes - so **no offline arbiter here resolves 0.005, and the ~390-study public LB resolves ~0.02**.
Plan fewer, bigger bets. Findings behind the rule: the consensus-subset idea is **refuted** (its labels are *less*
faithful to gold, 0.781 vs 0.850, because weak targets carry fold-local EB calibration that consensus discards);
error-reproduction by the blend is **not measurable** (all CIs cover 0.5); the blend's demonstrable gain is on the
*cleanest* cells (+0.0088 [+0.0055,+0.0122]), not the dirtiest; even unanimous three-voter cells are wrong 19 % of
the time vs gold, i.e. **the report-vs-image gap is irreducible by better extraction**. All seven arbiters rank
blend > arm A > eff > MVP; none reproduces the LB's 0.002 inversion. Tooling: `scripts/evaluate_runs.py`,
additive subset/bootstrap helpers in `metrics.py` (existing metrics verified digit-identical), suite 247 green.

- **Competitor review 2026-09-21** (`qa/competitor/analysis.md`, notebook `romantamrazov/rsna-knee-dinosaur-v5`,
  public LB 0.942 vs our 0.931): **the author trained none of it** - every scored checkpoint is a mounted public
  dataset from six other people, and his own training kernel's output is never read. His per-label weights are
  public-LB-tuned and he says so in the notebook ("on the strength of a public split that is a fraction of the test
  set... the likeliest place to give back points privately"); the scalars show three LB hill-climbs for +0.001
  total. Read 0.942 as ~0.93-0.94 real + 0.005-0.01 of public-split fitting. Their folds are report-md5, not
  site-grouped (their own EDA says this "does nothing about the scanner"), so their offline numbers are inflated -
  which is why they tune on the LB. **We are equal or better on**: 130 mm crop (identical constant), IPP ordering,
  uint8 cache, per-series (not per-slice) windowing, FOV padding, site-grouped folds, EMA, 384 px vs 336,
  fingerprint-gated loading. **Adopt (in order):** (A1) a 4th label voter - their public table measures gold 0.870
  vs our 0.850 and the rank-average of both **0.877, +0.0270 [+0.0148,+0.0381], P=1.000**, the only item clearing
  §4.2's bar; prefer our own second *local* model family for provenance (theirs is hosted-API-derived - allowed as
  public external data but get a host ruling). (A2) per-label attention heads: 12 queries over the slot x slice bag
  with an anatomical prior added to the attention logits - the structural difference vs all three of their arms.
  (A3) **sagittal laterality canonicalisation is a SLICE-ORDER REVERSAL, not an in-plane mirror** (`flip(dims=[0])`
  on `(n_slice,H,W)`) - medial/lateral lives on the through-plane axis, so our §2.3 ban on in-plane sagittal flips
  is correct and canonicalisation is safe where `side_conf >= 0.9`. (A4) per-target max/top-2 pooling over z-TTA
  for focal findings (a tear on two slices falls between samples; mean-pooling dilutes it) - bundle, not alone.
  **Reject:** per-label outer weights, the LB-ratcheted scalars, an 88-feature calibrator with no recorded fit set,
  best-epoch selection on 58 gold rows, report-md5 folds, denser inference sampling (our K sweep is on the flat
  part). **Do not build on third-party checkpoints**: winners' obligations require open-sourcing and defending the
  final model, and their RadImageNet encoder is CC-BY-NC-SA-4.0, a licence problem for a prize entry.

- **targets v1.1 built 2026-09-21** (`data/targets/v1_1/`, `qa/competitor/public_table.md`): 4th voter added.
  Public `pilkwang` table reproduces at rank-average **+0.0243 [+0.0139,+0.0351] P=1.000**; the state-mapped
  4-voter merge that `make_targets` actually consumes gives **+0.0149 [+0.0052,+0.0248] P=0.998** - clears §4.2.
  ~0.013 of the rank-average gain is lost translating their 3-level YES into our 2 cells (worth revisiting).
  Coverage is 4,406/4,407 and 57/58 gold, so every "gold-58" number against it is really gold-57.
  **Gold-leak test: NO leak** (1/5 flags, and that one is enrichment - our own llm_v1 shows the same 0.159 gap on
  those rows). **Their raw ACL AUC 0.997 is a granularity artefact of a 5-level score, not a leak (0.941 once
  state-mapped) - always run that control before calling a leak.** Merge machinery is now n-ary; `data/targets/v1`
  regenerates **bit-identical** under it; `_v3` is invisible to `metrics.voter_codes` so `consensus`/`strict` stay
  the same yardstick across target versions (test-pinned) - **judge v1.1 on consensus and gold, never weak-AUC**.
  Impact: 27.4 % of cells move, weight mass 0.695 -> 0.648 (watch this: the 3rd voter's disagreements cost 6.7 % of
  supervision). Prompt carry-forward ported as `base_qa` = `base` + the strict prompt's attribution/quoting
  paragraphs only, with the two clauses that are really *state* rules dropped. Suite 258 green.
  **Provenance decision (user, 2026-09-21): take the clean path** - our own second local extractor
  (`RedHatAI/gemma-3-27b-it-quantized.w4a16`, different lineage, 140+ languages, vLLM-native) is queued as screen
  `tv11`; the public table stays a measured fallback that would need a host ruling (rule 4.b bars *us* from calling
  a hosted API; their CSV is public external data but we will not stake the entry on a third-party reading).
  GPU 0 queue reordered to **armA8 -> tv11 -> stage9b** (labels are the measured dominant lever; stage 9's own
  analysis says its headroom is concentrated in 2 of 12 keys). Note `tv11` waits on screens named `armA8 stage9`,
  so the re-queued stage-9 waiter had to be renamed `stage9b` to avoid a mutual-wait deadlock.

- **TRAP: never score a live chain** (found 2026-09-21): `train_fold` rewrites `fold<k>/oof.parquet` and
  `metrics.json` **after every epoch**, so assembling OOF while a chain runs silently mixes finished folds with
  mid-training ones and nothing downstream detects it. An early two-fold read had arm C "losing" to arm A
  (0.9018 vs 0.9163) purely because fold 1 contributed an epoch-2 model. `qa/stage8b/compare.sh` now refuses to
  score a run whose screen is still alive; do the same in any new scoring path.
- **Arm D geometry, decided by measurement not by smoke AUC** (`qa/stage8b/cor_extent.py`, 600 `cor_fs` series):
  `view_params` crops about the image centre, so what matters is twice the larger centre-to-skin distance -
  centred width p5 98.9 / p50 123.2 mm. A centred crop keeps the MCL (10 mm inside the skin) in 6 % / **22 %** /
  **44 %** / 100 % of studies at 80 / 90 / 100 / 110 mm. **A 90 mm coronal crop would delete the MCL in 78 % of
  studies** - clipping the periphery was free for arm C's *central* sagittal meniscus view but is not for an edge
  structure like the MCL. Hence arm D uses **100 mm** (still a 1.3x zoom); `lat_oa` is a compartment and survives
  either. Candidate d2 (90 mm) is kept only to demonstrate the smoke sweep is silent on crop size at equal cost.

### 4.3 Results in 2026-09-22, and what they settle

Assembled 5-fold OOF (all scored on the v1 yardstick): mvp 0.8463/0.8034/0.8466 · eff **0.9080**/0.8388/0.8666 ·
arm A **0.9215**/0.8524/**0.8905** · armA8 (8 ep) **0.9240**/0.8534/0.8863 · arm C **0.9254**/**0.8576**/0.8878.

- **Gemma-3-27B as a 4th voter (clean provenance) - ACCEPTED.** `gemma3_merge_4v` +0.0118 [+0.0051,+0.0190]
  P=1.000 on 480 studies/57 gold; the public table's +0.0149 [+0.0052,+0.0248] is only marginally better, so we get
  ~79 % of the gain with labels that are ours to open-source. **Gemma alone is NOT better than Qwen alone**
  (+0.0029, P=0.654) - the whole gain is voter *diversity* (per-label correlation 0.906; lowest contusion 0.793).
  Targets built at `data/targets/v1_1_gemma3` (clean) and `data/targets/v1_1` (public table, fallback).
- **8-epoch schedule - a TIE on score, ADOPTED for cost.** weak +0.0025, gold -0.0042, both far below the 0.005
  resolution limit; but 2.07 h/fold vs 3.03 h = **32 % cheaper**. Use 8 epochs for every future convnext arm.
- **Stage-9 pseudo-labels - NOT BOUGHT.** One honest fold: weak +0.0042, gold -0.0128. Below resolution, and the
  leak-free 5-fold version needs ten leave-2-out sources (~20 GPU-h). Rejected under "fewer, bigger bets".
- **Targets v1.1 does not demonstrably improve the MODEL.** Arm C fold 0 retrained on v1.1: 6 of 7 arbiters rank
  v1.1 above v1, but every delta is < 0.005 (weak +0.0007, consensus +0.0022, strict +0.0054) and gold-58 moves
  -0.0455 on a CI 0.159 wide, i.e. noise in both directions. **Lesson: a +0.015 label-quality gain is too small to
  move the model measurably.** The earlier rules->LLM jump moved OOF +0.0265 because it was a far larger label
  change. Consequence: do not spend ~27 GPU-h retraining settled arms on v1.1 - instead train *new* arms on
  `v1_1_gemma3` so the label improvement rides along free.
  **[Corrected 2026-09-26, reviewer]** This is UNRESOLVED, not settled: it rests on one fold with 11 gold rows.
  Treat "a +0.015 label gain does not move the model" as a hypothesis, not a finding.
- **Arm C is a TIE as a replacement for arm A** (weak +0.0039, gold -0.0027 assembled, against +0.0075 on fold 0
  alone - fold-0 estimates overstate). Its value is as an **ensemble member** (fold-0 A+C was +0.0137); the
  assembled ensemble comparison is the number that decides the final entry.

- **main_v3 = arm A + eff + arm C -> public LB 0.933** (submission 56488491, 2026-09-23, dataset v5 + kernel v10,
  15 members, kernel-vs-local 1.78e-4), best so far. Assembled OOF: weak **0.9341** / strict 0.8685 / consensus
  0.8900 / gold **0.9044**; vs arm A +0.0126 weak [+0.0098,+0.0154] P=1.000 with strict/consensus/gold all up -
  §4.2 clause 2. A8 is a near-duplicate of A (rho 0.922; A8+E+C vs A+E+C +0.0003 = tie). Kaggle runtime ~3.4 h
  (~4.2 h with margin; main_v3 is 2.1x main_v2 per study), so the 9 h budget holds roughly 1-2 more arms.
  Weakest labels in the blend: lat_oa 0.8729, mcl 0.8886. Export hazard still open: `export_kaggle.py` accepts
  any fold with a bundle.json, and smoke runs under `runs/arm*_sweep_*` hold real bundles.
- **Licence scare, 2026-09-25 - FALSE ALARM, read the rules first.** The developer audit flagged
  `convnextv2_tiny.fcmae_ft_in22k_in1k_384` as CC-BY-NC-4.0 (true, per its HF card) and I briefly redirected arm D
  and the per-label-head experiment to an Apache backbone. The rules text (read directly via the Kaggle API) says
  **"6. WINNER LICENSE TYPE CC-BY-NC 4.0"** - the same licence - and **rule 5.a.4**: "In the event that input
  data or pretrained models with an incompatible license are used to generate your winning solution, you do not
  need to grant an open source license ... for that data and/or model(s)." So NC-licensed pretrained weights
  (ConvNeXt-V2 FCMAE, RadImageNet) are **usable**; the change was reverted within the hour. ND ("no derivatives")
  licences remain a real problem for released derivative weights; the host has banned KneeCoT specifically.
- **Hosted LLM APIs ARE ALLOWED - the §2.1 "rule 4.b" premise was wrong.** Host post, topic 733965, 2026-08-09:
  "Use of commercially hosted LLMs and other external inference services is permitted ... submitting Competition
  Data, including report text, to an external LLM or API for inference ... (for example, extracting labels from
  reports) will not, by itself, be considered prohibited PRIVATE SHARING" (rules §2.6.b). The public `pilkwang`
  table (Claude-API-derived) and any API pass of our own are legitimate. Lesson for both scares: **check the
  primary rules text and host posts before acting on a rules premise** - two agents disagreed, and only reading
  the source settled it.

### 4.4 Audit 2026-09-25 and what followed

Six-role audit in `qa/audit_2026-09-25/` (1 researcher, 2 analyst, 3 architect plan, 4 developer, 5 tester,
6 reviewer). Headlines: more arms of the same kind have saturated (+0.0077, +0.0049, +0.0007); A+E+C is the honest
optimum of existing members; the model beats its own labels on gold by 0.07; 71 % of the gold-visible loss sits in
lat_oa, synovitis, pf_oa, lat_men, contusion; laterality is not an error source (flip canonicalisation < 0.001);
the tester verified main_v3 end to end (275 tests; kernel vs local 1.78e-4; time guard fires correctly; compressed
DICOMs decode identically). Only one code file had drifted local vs server (the generated notebook); reconciled to
the live kernel v10 copy (md5 e61dc63c) on 2026-09-26.

- **Pseudo-label round (user decision 2026-09-25)**: teacher = main_v3 OOF blend, targets `data/targets/v3_pseudo`
  = 0.5 x v1_1 + 0.5 x teacher on all cells incl. silent, silent under-reported cells re-weighted 0.1 -> 0.5.
  Clean pre-check before any student trained - agreement of the *targets* with gold: v1_1 0.845, teacher 0.904,
  **mix 0.915 (+0.070 [+0.045,+0.097] P=1.00)**; synovitis 0.635 -> 0.799, effusion 0.774 -> 0.940, fracture
  0.803 -> 0.926; lat_oa flat. Judge the student on gold-58 + consensus + a paired LB probe; its weak OOF is
  contaminated by construction (teacher OOF carries fold-k information), gold rows are in no training set.
- **Claude label pilot (user decision: run via the Claude Max plan through Claude Code subagents, no paid API)**:
  481-study subset, 5 Sonnet workers, 481/481 parsed, `data/labels/claude_subset.parquet`, prompts identical to
  the other voters (`qa/claude_labels/`). Claude alone as a 4th voter: +0.0125 [+0.0059,+0.0199] P=1.000 -
  interchangeable with Gemma (+0.0118) and the public table (+0.0149). **Marginal value on top of
  {ours, gemma3, public}: +0.0043 [-0.0010,+0.0098] P=0.945 - below §4.2, so the full 4,407-report pass was NOT
  run** (`qa/claude_labels/marginal_gain.json`). Extractors share blind spots (lowest correlation with ours on
  fracture, contusion, lat_oa for all three), so each extra voter buys less. Rules basis for using a paid
  subscription: §6.b explicitly accepts "a small subscription charge ... such as Gemini Advanced"; the task itself
  costs a few dollars on any provider's API, so it is reasonably accessible. For the winners' write-up, record the
  model (Sonnet), the prompt files and the raw outputs.

### 4.5 Plan as of 2026-09-26 (architect's plan with the reviewer's required changes applied)

Sources: `qa/audit_2026-09-25/3_architect.md` (plan) and `6_reviewer.md` (verdict: "not fit as-is, fit after 8
changes"). Changes applied:
- **Paired noise, not level noise.** Two submissions scored on the same public studies differ with a noise of only
  ~0.002-0.003 between related ensembles (~0.006-0.010 for an effnet vs its retrained twin); gold-58 differences are
  paired too (+-0.007-0.009 between ensembles, not +-0.027). Gates use paired differences.
- **Pseudo-label bet (bet 1).** The effnet student E_p is a KILL-ONLY check (its teacher beats it by +0.026 OOF, so
  it can "pass" on distillation alone): kill the lane only if E_p's paired gold-58 is below eff_v1's by > ~0.009.
  The ConvNeXt students A_p (arm A recipe, 8 ep) and C_p (arm C recipe, 8 ep) are trained regardless, queued
  behind the running chains. **Decision: main_v4 = A_p + E_p + C_p vs main_v3 on pooled paired gold-58 + one paired
  LB probe; accept at >= +0.004 with gold >= -0.009.** Consensus is removed from every promote decision (blind to
  silent cells, penalises the intended relabelling, flattered by the teacher leak).
- **Stopped/paused:** the Claude-voter lane (marginal +0.0043, below the bar; never again on the shared quota);
  efficiency decode engineering and the efficiency-slot branch (rank 478 cannot reach a prize); arm N triage paused
  2026-09-25 23:48 to give GPU 0 to the main bet - relaunch command in `qa/stage11/` when a GPU is idle.
- **Final selection:** slot 1 = main_v4 if accepted, else main_v3; slot 2 = the other main entry (a second main
  pick is worth only +0.0002..+0.001 expected private AUC, but the efficiency track is out of reach). Pending user
  confirmation. Select 56488491 in the UI now as insurance (user action; no CLI for it).
- **Owed measurements:** main_v3 and eff_v1 hidden runtimes (visible only in the Kaggle UI); z-TTA's value.
- **Engineering before the next export:** git on the server with a `main_v3-shipped` tag, export guard (live /
  incomplete / smoke folds), liveness guards in scoring tools, `Fat_Suppression` tie-break in slot routing (host
  declared the flags may differ in test; must be byte-identical on train), then inference robustness by 10-06.
- **GPU capacity is not the constraint** (~1,100 GPU-h available to the freeze vs ~100-150 planned); Kaggle's 9 h
  runtime is. Candidate for idle GPU time after main_v4: full-data retrains / extra seeds of the final arms, checked
  by a paired LB probe. Every chain must have its successor queued (2.5 GPU-days were lost to idling once).

- **Engineering lane landed 2026-09-26** (`qa/eng/changelog.md`): the server tree is now a **git repo** (no
  remote); baseline `f789876` tagged `main_v3-shipped` (notebook md5 e61dc63c verified first); `main` fast-forwarded
  to `8185651` = export guard (refuses live runs, folds with last.pt or not at the final epoch, missing folds,
  smoke/sweep runs), liveness guards in predict_oof/evaluate_runs/compare_ensemble (a chain's own processes are
  excluded as ancestors - checked against the live queue before merging), git sha in every bundle, and a
  `remote.ps1 sync` drift check that refuses to push over newer server files (`drift` action = check only).
  Branch `eng/p0` also holds `9a4946d` (Fluid_Sensitive != Fat_Suppression routing: mismatched series rank after
  matched ones; byte-identical slot assignment on all of cache v1) and `75d8c5e` (per-study 600 s preprocessing
  timeout, per-study exception -> prior row, dead consumer thread -> prior rows, exit 3 above 10 % prior rows).
  **These two change inlined notebook code: merge them at the main_v4 release together with the notebook rebuild
  and a kernel-vs-local check.** Work was done in a separate worktree so running chains never saw half-edited
  files. 322 tests + 1 xfail (notebook freshness, strict under RSNA_KAD_RELEASE=1).

- **Pseudo-label round 1 RESULT, 2026-09-26** (`qa/main_v4/compare_v4_vs_v3.txt`, 2000 paired bootstraps):
  single models on gold-58: Ep 0.8999 (vs E 0.8666), Ap 0.9078 (vs A 0.8905), Cp 0.9197 (vs C 0.8878).
  **main_v4 = Ap+Ep+Cp: gold-58 0.9179 vs main_v3 0.9044, paired +0.0135 [-0.0001,+0.0279] P=0.974** - clears the
  gold side of §4.5's rule; LB probe pending. Membership: arm D adds nothing (-0.0019); old+new mixed is worse than
  new alone (-0.0031) so students replace members; Ep is a tie inside the blend (Ap+Cp -0.0012) but kept as the
  cheapest member. Weak/strict rise ~+0.020/+0.025 but are contaminated for pseudo runs - never quote them as
  evidence. Every label rose on weak OOF incl. the two worst: lat_oa 0.873 -> 0.906, mcl 0.889 -> 0.924.
  **Lesson: pseudo-labelling was the lever; our stage-9 rejection of it was an artefact of judging with a metric
  that could not see it.** Round 2 (teacher = round-1 students) launched the same night.
- **First LB confirmation, 2026-09-26: efficiency probe E_p = public LB 0.930** (submission 56588465, dataset
  `rsna-kad-weights-effp` + kernel `rsna-kad-effp`, kernel-vs-local 2.19e-4, K16 no-TTA - identical inference to
  eff_v1) vs eff_v1 **0.913**: **+0.017**, beyond the ~0.006-0.010 paired noise of an effnet vs its retrained twin.
  Forecast made before scoring was ~0.928 (0.922-0.934). A single 256 px model now nearly matches the old 15-member
  main_v3 (0.933). main_v4 (submission 56588582) forecast ~0.941 (0.936-0.946), still scoring.
- **main_v4 = public LB 0.943** (submission 56588582, 2026-09-27; dataset `rsna-kad-weights-v4` + kernel
  `rsna-kad-main-v4`, notebook md5 6201e15f built from 1b769e0, kernel-vs-local 1.31e-4). +0.010 over main_v3, inside
  the forecast. Both pseudo-label forecasts landed in range, so the gold-58 -> LB mapping (~0.5-0.8 of the paired
  gold gain transfers) is holding. We now sit level with the 0.943 public-fork pile-up using only our own models.
- **Pseudo-label round 2 - NEGATIVE, stopped 2026-09-27** (`qa/main_v4/round2_vs_round1.txt`): teacher = round-1
  students. Paired gold-58: Ep2 - Ep -0.0002 (tie); Cp2 - Cp -0.0050 [-0.0133,+0.0031]; Ep2+Cp2 0.9114 vs Ep+Cp
  0.9188. A teacher built from pseudo-trained students re-teaches their own opinions (confirmation bias), so a second
  round reinforces errors. Arm A's round-2 student was stopped at fold 3 (its `last.pt` remains, so the export guard
  refuses it). **Do not iterate pseudo-labels on our own students again;** a further round would need a genuinely
  independent teacher/label source. (Scoring note: the liveness guard matches runs by name, so a finished run whose
  chain script is still alive is refused - verified E_p2 complete by hand and scored a renamed frozen snapshot.)
- **z-TTA - DROP from the final ensemble** (`qa/ztta/arbiters.md`): main_v4 blend with vs without z-TTA, paired
  gold-58 +0.0005 [-0.0021,+0.0034] P=0.35 - below the 0.002 rule. Frees ~1.7 h of Kaggle runtime for members.
- **Stacking - REJECTED, measured 2026-09-27** (`qa/stacking/stack_test.py`, `result.json`): 10 finished models (A, A8,
  E, C, D, Ap, Ep, Cp, Ep2, Cp2), stackers fitted with nested folds on v1 weak targets, judged on gold-58 paired vs
  main_v4 equal (0.9179): equal-10 -0.0037; stack-10 global weights -0.0037; stack-10 per-label LR with strong
  shrinkage -0.0052; stack-10 per-label LR -0.0071; stack of main_v4's 3 per-label -0.0025. **All five lose and
  loss grows with stacker flexibility** - the meta-learner fits the report-derived labels (inflated for pseudo runs),
  not the image-derived gold. Keep equal weights over a measured member set; spend freed runtime on seeds /
  full-data retrains of the main_v4 recipes, not on more or weighted members.
- **Architecture heads - CLOSED, 2026-09-28** (`qa/stage11/heads_final.txt`, all on v3_pseudo, arm C's recipe, paired
  gold-58). Per-label attention, depth 1 (triage winner = the variant WITHOUT the anatomical prior): 0.9201 vs C_p
  0.9197 (+0.0005, tie); replacing C_p in main_v4 -0.0004; as a 4th member +0.0021 [-0.0026,+0.0075] P=0.78 - not
  demonstrable, below the bar. Deep transformer head, depth 4, pre-norm, stochastic depth 0->0.15: assembled 0.8549
  because **fold 3 diverged** at epoch 1 (the first epoch at peak lr after warmup): loss 0.671 -> 0.689 ~ ln 2 and
  AUC ~0.5 for the rest of training; its four healthy folds scored about the same as C_p's. Rejected as unstable
  with no evidence of gain; a rescue (longer warmup, lower head lr) would cost ~10 GPU-h for an expected tie.
  **Conclusion: the pooling/fusion head is not the bottleneck at depth 1 or 4.**
- **Seed ensemble + full-data retrains, 2026-09-29** (`qa/full/score_gold.py/json`, gold-58 paired vs main_v4
  0.9179): C seed ensemble (Cp + 4 seed/HP variants, averaged, never picked) 0.9255 (+0.0076 [-0.0003,+0.0160]
  P=0.97); but the PRE-DECLARED test "main_v4 with C slot = seed ensemble" is a tie (+0.0005). Full-data single
  models beat their fold counterparts (Afull 0.9210 vs Ap 0.9078; Efull 0.9110 vs Ep 0.8999; Cfull 3 seeds
  0.9239). main_v5 full-data arm-balanced (Afull+Efull+Cfull) 0.9239 (+0.0061, P=0.87). **After pseudo-labelling
  the arms converged (rho Ap-Cp 0.925) and C alone now matches or beats every blend** - but "C only" was found
  by scanning ~9 candidates on the same 58 studies (winner's curse), so it is a post-hoc hypothesis to be tested
  on the LB, not a conclusion. Caveat: full-data targets are calibrated on all 58 gold (per-label constants, not
  per-study) - a small optimistic tilt in full-data gold scores; the LB is the clean judge. Plan: two pre-declared
  paired LB probes vs main_v4 - A `main_v5_bal` (Afull x3 + Efull x3 + Cfull, arm-balanced) and B `main_v5_C`
  (Cfull only, 4-8 seeds). Release hazard: full-data bundles carry the head-experiment config keys, so the release
  branch merges `full-data` + `heads` and rebuilds the notebook; dry run first.
- **main_v5 dry run PASSED 2026-09-29** (branch `release/v5` = main + full-data + heads, merge 8a00d16; rebuilt
  notebook e52fc80, md5 40a44ec6; built in `.worktrees/release_v5` so running chains were untouched; `src/` and
  `scripts/` byte-identical to the tree the full-data models trained from). Loader accepts the head-experiment keys,
  0 prior rows, Spearman 0.988 vs main_v4 on the public studies. **Two exporter bugs found:** (1) the fallback prior
  for full-data runs was averaged over their 58 gold rows (enriched prevalence: ACL 0.376 vs 0.141) - must come from
  all 4,407 OOF rows; (2) `export_kaggle.py` writes weight 1.0 for every member, so arm balancing silently becomes
  seed-count weighting. Worked around by `qa/main_v5/finalize_ensemble.py` (arm-balanced weights, main_v4's prior);
  the proper fix belongs in `kaggle_export.py`. Projected hidden runtime: balanced ~1.9 h, C-only ~1.4 h (vs main_v4
  ~3.4 h projected), thanks to one model per full-data seed and no z-TTA.
- **main_v5 probes, 2026-09-30:** **B `main_v5_C` (6 Cfull, equal) = public LB 0.948 - new best** (submission
  56687125, ~1.2 h projected); A `main_v5_bal` (3 Afull + 3 Efull + 6 Cfull, arm-balanced) = 0.945 (56687124).
  Forecasts made before scoring: B ~0.945, A ~0.946 - both in range. B - main_v4 = +0.005 (beyond paired noise);
  B - A = +0.003 (edge of LB resolution, same direction as gold). The post-hoc "C only" hypothesis was confirmed on
  the independent LB. **After pseudo-labelling the arms converged and the weaker arms (E especially) dilute C.**
  All 12 C full-data models done (8 seeds 0.917-0.923, 4 HP variants 0.918-0.923 gold). New test launched: arm A's
  backbone on arm C's five views (`configs/main_armA5v_pseudo_full.yaml`, 3 full-data seeds) - is the 90 mm
  sagittal view, not the backbone, what makes C strong? Next release = one probe of the likely final composition
  (C x12, + A5v x3 if its gold matches C); a C6 -> C12 probe alone would be below LB resolution.
- **A5v (arm A backbone on arm C views) - NOT included, 2026-09-30:** full-data gold 0.9161/0.9174/0.9142 (mean
  ~0.916) vs Cfull mean ~0.920 and Afull (4 views) 0.916-0.921. The 90 mm view did not lift ConvNeXt-small to C's
  level, so **C's strength comes from the ConvNeXt-V2/FCMAE backbone, not the view alone** - the hypothesis that
  launched A5v was wrong. Follow-up running: `configs/main_armcb_pseudo_full.yaml` = C recipe on
  convnextv2_base.fcmae_ft_in22k_in1k_384 (smoke: fits, 19.5 GB peak; est. ~10 h per full-data seed, ~3x tiny
  inference cost). Seeds 2026/2126 on GPU 1, 2226 on GPU 0. main_v6 (12 Cfull = 8 seeds + 4 HP variants) released
  as a verification probe of the likely final composition; its expected gain over main_v5_C is below LB resolution.
- **main_v6 (12 Cfull) = public LB 0.947** (submission 56710956, 2026-09-30) - a tie with main_v5_C 0.948, as forecast
  (C6 -> C12 is below LB resolution; 6 of the 12 members are main_v5_C's). Projected hidden runtime ~2.2 h.
- **ConvNeXt-V2-base - a TIE, not shipped, 2026-09-30:** full-data s2026 gold 0.9204 vs tiny seeds mean ~0.920; s2226
  at 0.920 after 7 of 8 epochs; ~4.5 h per seed at 120-132 img/s (not the ~10 h estimated). 3x the inference cost for
  no measured gain. **The pipeline has plateaued at ~0.920 gold per model / 0.947-0.948 public LB:** seeds, backbone
  size, heads, stacking and a second pseudo round all land at or below it. The remaining lever we cannot produce
  ourselves is genuinely independent models (team merger, deadline 2026-10-15; user decision).
- **Pipeline B (user request 2026-09-30): an independently built model family to merge with ours**, the in-house
  substitute for a team merger. Literature scan `qa/sota_2026-09/research.md`: image-report contrastive pretraining
  is NOT supported at our scale (positive results use 7k-200k pairs, none multilingual; text supervision helps only
  well-described findings, not under-reported ones) - reports are best used as more, finer labels; different
  pretraining objectives decorrelate errors far more than seeds (ICLR 2022); missing-label losses (CVPR 2021-22) fit
  "not mentioned != negative"; no usable public knee-MRI foundation model (OAI-trained, research-only or no weights);
  3-D is more different, not stronger. Proposed B: DINOv3 ViT-B/16 @288, all slices + mean pool + 1-layer slice
  transformer, extra coronal/axial-PF crops, auxiliary findings, missing-label loss, own teacher for its pseudo-labels
  (never A's). Research's estimate: 40-50 % chance the merge clears the bar, +0.002-0.006 LB if it does. Design +
  licence verification + T4 budget: `qa/sota_2026-09/pipeline_b_design.md` (architect).
- **Pipeline B BUILT 2026-10-01** (branch `pipeline-b`, 0ca784d/de34fd6; design `qa/sota_2026-09/pipeline_b_design.md`):
  DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`, DINOv3 licence permits derivatives/redistribution under its own
  terms) @288 px from the 384 cache, 5 views incl. 90 mm sagittal and 110 mm coronal, 12 + 24 auxiliary outputs (aux
  from a new Qwen pass, weight 0 where unmentioned), large-loss rejection for under-reported keys in round 0, LLRD,
  MRI intensity augmentation (shared code from `mriaug` dc4ec67), B's own teacher (builder refuses non-`runs/b_*`).
  **Kaggle image timm = 1.0.26, DINOv3 builds there** (probe kernel; no wheel shipped, A unaffected; server trains on
  1.0.29 - the kernel-vs-local check must cover the gap). T4: one study x one B member ~0.45 s fp16 -> ~5 min per
  member on 2xT4 for ~1,300 studies. 432 tests green. Six chains armed behind `qa/A_DONE` (user's order: pipeline A's
  pending tests - pairwise loss, multi-crop/rotation TTA, MRI augmentation - finish first), with automatic G0/G1
  gates and pre-declared fallbacks (ConvNeXt-S DINOv3). Merge rule fixed: 0.5 x mean(A) + 0.5 x mean(B). Release
  prep (per-pipeline weights, proper prior fix, DINOv3 licence notice, notebook rebuild) assigned to the release
  owner on `release/v7`.
- **Final-selection recommendation (pending user):** slot 1 = best C-only entry; slot 2 = `main_v5_bal` (0.945) as a
  structural hedge - a different member composition protects against "C only" being partly a public-split fluke.
- **Process miss 2026-09-29:** GPU 1 idled ~10 h (01:28-11:30) because its queue ended with no successor. Rule:
  when queuing a chain, always queue its successor in the same step. **Repeated the same evening** (GPU 1 idle
  19:35-22:31: `full_gpu1c` was queued without a successor, hours after writing the rule). Every queue must now end
  in a named follow-on job; `qa/full/after_cfg.sh <wait-screen|-> <gpu> <config>:<seed> ...` is the generic runner.
- **Full-data set as of 2026-09-29 22:30:** Afull s2026/2126/2226 gold 0.921/0.916/0.917; Efull 0.911/0.900/0.909;
  Cfull s2026-2526 0.922/0.922/0.919/0.919/0.922/0.920 (s2626/2726 still training). Queued: full-data versions of
  the four seed-ensemble HP variants (`configs/main_armc_p_hpe{2..5}_full.yaml`, seeds 3026-3326; only run_id and
  targets_version differ from the 5-fold configs) - more arm-C diversity, averaged, never picked. Probes A
  (`main_v5_bal`, 12 members) and B (`main_v5_C`, 6 C members) released from the 6 landed C seeds.
- **Liveness-guard false positives (engineering follow-up):** the guard matches a run by its name appearing in any
  live process's command line or referenced script, so finished runs mentioned by *other* queued chains (E_p2 by the
  round-2 chain, C_p by the seed-ensemble chain) are refused. Worked around with renamed frozen snapshots after
  verifying completion by hand (ALLDONE in the log, no last.pt, final epoch). Fix: treat a run as live only if a
  live `train.py`/`predict_oof.py` process is writing it (its config's run_id), not if a script merely names it.
- **main_v5_C + crop TTA probe, 2026-10-01 = public LB 0.947** (submission 56751245; main_v5_C 0.948) - a tie, below LB
  resolution, so crop TTA is not adopted on this evidence (gold-58 had it at +0.0027, P=0.94). Same 6 Cfull members,
  weights and prior as 56687125; the only change is a second view per member with every field x 115/130, averaged
  equally (qa/tta2 "base+crop"). Opt-in ensemble.json `view_scales`, default off, on `release/v7` 8f89227 (notebook
  d8457bf, md5 d80071d2b5c877d00a4d917ddd1aa631); dataset `rsna-kad-weights-v5c-crop` v1 (CC BY-NC 4.0, member files
  byte-identical to v5c) + kernel `rsna-kad-main-v5c-crop` v1. Checks: base-only bit-identical to the e52fc80 notebook
  at equal thread count; probe = equal-weight mean of both views (7e-7); kernel-vs-local 5.9e-4. Hidden run <= 2 h
  20 min (projected 2.3 h, 2x main_v5_C's GPU time).
- **Pipeline B G2 FAILED 2026-10-02 -> B stopped** (design §7; no LB probe). Merge = 0.5 x main_v5_C's 6 Cfull +
  0.5 x B r1 full seeds 2026-2326: gold-58 0.9248 vs A-final 0.9254, paired delta -0.0004 [-0.0101, +0.0088],
  P=0.46 (bar +0.004). B seeds 0.900/0.908/0.903/0.904 (mean 0.9037, its own bar 0.900 met), A seeds ~0.920.
  G1 had passed on round 0 (merge +0.012 over C_r0, mean rho 0.77); after pseudo-labelling both pipelines the
  gold-row rho is ~0.90 and the diversity gain is gone - the same convergence that made main_v5_C beat the 3-arm
  blend. Evidence `.worktrees/pipeline_b/runs/b_gate/G2.json` (script `qa/pipeline_b/adhoc/g2.py`). B seeds 2426/2526
  are not pre-declared and are not used to re-test. release/v7's merged path stays unused.
- **PRE-DECLARED 2026-10-03: max-pool residual and nnPU tests** (branch `maxpool-nnpu` 69a8b8e, worktree
  `.worktrees/maxpool_nnpu`; ideas from an external review, user-approved). Both are main_armc_pseudo_full.yaml
  with one switch, 3 full-data seeds each (2026, 2126, 2226), paired with the existing Cfull runs of the same seeds.
  - *Max-pool* (`configs/main_armc_pseudo_full_maxpool.yaml`, `model.mil_maxpool_residual: true`): per-slot pool
    = gated attention + g x max over the K triplets, g learnable, init 0. Hypothesis: 1-2 slice focal findings
    (lateral meniscus tears) are averaged away by attention.
  - *nnPU* (`configs/main_armc_pseudo_full_nnpu.yaml`, `loss: bce_nnpu`): on the 7 under-reported labels the
    report-silent cells of trusted reports get the non-negative PU risk pi R_P+ + max(0, R_U- - pi R_P-) instead
    of soft BCE; P = affirmed (mild only for synovitis/fracture/contusion, calibrated P(gold+|mild) >= 0.5);
    pi = calibrated P(gold+|silent) of v3_pseudo_full (synovitis 0.310, fracture 0.103, contusion 0.105,
    baker 0.056, med_oa 0.082, lat_oa 0.100, pf_oa 0.183); R_P- a detached EMA 0.98 (a 4-study batch often has
    no affirmed cell); plain clamp; PU term weighted at the silent cells' share of total target weight, 0.154;
    mixup steps stay BCE. All fixed from the targets before any gold number. Hypothesis: "silent" is unlabelled,
    not 0.5 label + 0.5 teacher.
  - **Judge** (each separately): `qa/full/compare_pw.py --new <run>_s{}`, 3-seed mean vs Cfull s2026/2126/2226 on
    gold-58, paired bootstrap 2000. **Accept iff delta >= +0.004 with P >= 0.95, else reject.** Per-label deltas
    are reported, not acted on. Outputs `qa/maxpool/compare.txt`, `qa/nnpu/compare.txt`.
  - **RESULT 2026-10-03: both REJECTED.** Max-pool: 3-seed gold 0.9233 vs 0.9239, delta -0.0006 [-0.0063,
    +0.0051] P=0.40 (lat_oa -0.015, lat_men -0.009). nnPU: 0.9199 vs 0.9239, delta -0.0040 [-0.0141, +0.0056]
    P=0.20; synovitis - its main target - fell 0.812 -> 0.780 and lat_men -0.022, while contusion +0.024 and
    fracture +0.011 (not acted on). The 0.5 label + 0.5 teacher mix handles "silent" cells better than nnPU.
- **Two LB probes, 2026-10-03 (user-approved, one each; final selections unchanged): both tie main_v5_C 0.948.**
  (1) **`main_v5_C` 12 seeds = public LB 0.947** (submission 56803029): Cfull s2026-3126, 1/12 each, main_v5_C's
  prior, no TTA, K 24 (gold 0.9254 vs 0.9255) - meets the pre-declared bar (>= 0.947), so it is a slot-1
  candidate; 12 seeds buy no measurable gain over 6, only robustness. (2) **main_v5_C + 3 max-pool + 3 nnPU =
  public LB 0.948** (submission 56804636; gold 0.9251) - the two training changes rejected above on gold neither
  help nor hurt inside the blend. Both on `release/v7` (notebooks d80071d2 / 72dddd56; max-pool residual ported from
  maxpool-nnpu 69a8b8e as an opt-in per-member config flag, `1053e52` + notebook `a6afed8`; `--prior-of-bundle`
  `73506a7` keeps main_v5_C's prior exactly). New private CC-BY-NC datasets/kernels `rsna-kad-weights-v5c12` /
  `rsna-kad-main-v5c12` and `-v5c-mpnnpu`. Checks: first 6 members reproduce main_v5_C (CSV rounding); 12-member
  output = mean of the two halves (1e-16 at full precision); max-pool members' logits bit-identical to the
  branch's model; kernel-vs-local 3.0e-4 / 3.4e-4; hidden runs <= 2 h 11 min / <= 2 h 00 min.
- **main_v5_C + 0.2 B_r0 on the weak labels = public LB 0.947, 2026-10-04** (submission 56808088; user-approved
  despite failing the +0.004 gold gate, gold 0.9263 vs 0.9255) - a tie with main_v5_C 0.948. Output = main_v5_C on 8
  labels; synovitis/lat_oa/pf_oa/lat_men = 0.8 x main_v5_C + 0.2 x mean of the 10 pipeline-B round-0 fold models
  (DINOv3 ViT-B/16 @288, K 16). New opt-in per-label group weights (`--label-group-weights`, ensemble.json member
  `label_weights`, `release/v7` d56a529, notebook e935d0b md5 27d5542b); untouched labels bit-identical to main_v5_C
  locally, blend exact at full precision (1.7e-16). **First B-containing dataset `rsna-kad-weights-v5c-bweak`: Kaggle
  accepted licence "other"** (metadata reads back `other` + description), with LICENSE-DINOv3.md + NOTICE. Kaggle
  image now torch 2.11 (2.10 on 09-29); A reproduces main_v5_C's kernel output to 4e-5. Kernel-vs-local: ensemble
  3.3e-4; **B members (derived on the 4 weak labels) 8.9e-4 max** (fp16 ViT on T4 + timm 1.0.26 vs fp32 CPU + 1.0.29)
  - no sign of a timm-version break. Hidden run <= 2 h 03 min.
- **main_v5_C + silentq synovitis probe, 2026-10-05: submission 56837972 ERRORED (no score)** - user-approved, one
  submission. 11 labels = main_v5_C; synovitis = mean of 3 full-data arm-C models trained with report-silent
  synovitis cells on the teacher only (gold-58: 3-seed mean 0.9258 vs 0.9239, synovitis 0.812 -> 0.838). Bundle
  `rsna-kad-weights-v5c-silentq` (CC BY-NC, 9 members, A=1,S=0 + synovitis:A=0,S=1), kernel `rsna-kad-main-v5c-silentq`
  v1, notebook e935d0b md5 27d5542b - the same notebook that scored the 16-member B probe on 10-04. Every gate
  passed before submitting (11 labels bit-identical to main_v5_C, synovitis = mean of S at 5.6e-17, kernel-vs-local
  3.3e-4, A reproduces main_v5_C's kernel to 5e-5, projected ~1.9 h). The submission sat PENDING and went COMPLETE
  with a BLANK public score at ~09:51 UTC, 9 h 11 min after submitting - the signature of a hidden-run failure at
  the 9 h limit (timeout/hang, or a CPU fallback hitting the 8.25 h guard and the >10 % prior-row exit 3), not of
  our code: nothing differs from the scored 10-04 probe but 3 same-architecture members. Hidden-run log not
  available via CLI; the UI's error label decides. Not resubmitted (coordinator's rule: report before resubmitting).
- **Inference hang protection, 2026-10-05** (Kaggle UI: 56837972 = "Notebook Timeout"; `release/v7` e6065ca, notebook
  a3c5cc1 md5 b328ad1a). Diagnosis: a slow run cannot time out (the 8.25 h wall guard ends it by ~8.3 h, a CPU
  fallback then exits 3 = "threw exception"), so something blocked where the guard never runs. Unbounded waits
  found: the driver's put to a device thread (loops until the thread raises - one stuck in a device op never
  does), put(None)/join() in its finally, Pool.join on a worker in uninterruptible I/O (stalled /kaggle/input
  mount), interpreter exit joining such a worker after submission.csv was written, main-thread I/O before the
  guard, the prelude's pip. Most likely trigger: a wedged GPU op or stalled input mount on the scoring machine.
  Fixes (all bundles, output bit-identical when nothing hangs): an independent watchdog thread (8.5 h hard, 30 min
  without progress) writes finished studies + prior and exits 0 (WATCHDOG FIRED in the log, run_report.json);
  heartbeat-bounded waits on device threads (stalled rows = STALL_ERROR, excluded from the prior alarm); decode
  pool restarted after a preprocessing timeout; pool shutdown time-bounded; script ends with os._exit; pip 900 s
  timeout; no CUDA device -> first member per group at K <= 8 (NO CUDA DEVICE); any exception -> fallback file,
  exit 0. Tests `tests/test_watchdog.py` (11, incl. the generated notebook rescued from a hang outside the driver);
  strict suite green. CPU re-verification: main_v5_C, the silentq bundle and main_v5_C12 reproduce the previous
  notebook bit for bit (0.00); on Kaggle the hardened kernels equal their old versions exactly (0.00).
- **main_v5_C + silentq synovitis, hardened resubmission = public LB 0.948, 2026-10-05** (submission 56854249,
  kernel `rsna-kad-main-v5c-silentq` v2 on the unchanged dataset; hidden run <= 1 h 51 min). Ties main_v5_C 0.948
  and meets the pre-set bar (>= 0.948): a slot-1 candidate. The hidden run's log is not retrievable; a watchdog
  firing would have left prior rows and cost score, and the run finished in normal time, so there is no sign it
  fired. The 56837972 timeout was therefore not caused by the bundle. **Hardened final candidates ready, not
  submitted:** `rsna-kad-main-v5-c` v2 (main_v5_C, dataset v5c) and `rsna-kad-main-v5c12` v2 (12 seeds, dataset
  v5c12), both notebook b328ad1a, outputs identical to their v1 on Kaggle; resubmitting them is the user's call.
- **t70 heavier teacher, 3 seeds = public LB 0.944, 2026-10-06** (submission 56874956; user-approved probe). Full-data
  arm C on v3_pseudo_full_t70 (0.3 label + 0.7 teacher), s2026/2126/2226, 1/3 each, main_v5_C's prior, no TTA, K 24;
  gold-58 0.9233 vs 0.9239 (a tie). Dataset `rsna-kad-weights-t70` + kernel `rsna-kad-main-t70` v1, hardened notebook
  a3c5cc1 (md5 b328ad1a): output = mean of the 3 members (1.1e-16), watchdog armed and silent locally and in the
  kernel run, kernel-vs-local 4.2e-4 (3 members average out less fp16 noise than 6), hidden run <= 1 h 11 min.
  0.004 below main_v5_C 0.948; the 3-vs-6-seed difference explains at most ~0.001 (6 -> 12 seeds was LB-neutral), so
  roughly -0.003 is t70's - still inside the public LB's ~+-0.01 noise, consistent with the gold tie. Not a candidate.
- **PRE-DECLARED 2026-10-04: silent-cell teacher-only targets ("silentq", external review idea #3).**
  - *Hypothesis:* a v3 report-silent cell of an under-reported finding has weight w on target 0.5 r + 0.5 q;
    BCE is linear in the target, so that is 0.5 w BCE(r) + 0.5 w BCE(q) - pseudo-labelling raised the
    "report silent" prior r from weight 0.1 to ~0.25. Removing it should help the findings radiologists omit.
    (Context: review idea #1, main_v5_C + 0.2 x pipeline-B round 0 on the 4 weak labels, scored gold 0.9263 vs
    0.9255, delta +0.0009 P=0.66 - failed the gate; synovitis +0.030 there, not acted on. qa/full/prepseudo_blend.*)
  - *Change:* targets `data/targets/v3_pseudo_full_silentq` (builder `qa/silentq/make_targets_silentq.py`): for
    synovitis, lat_oa, pf_oa, every cell whose weight v3 raised via absent_under_reported (report state "absent";
    parse-fail/postoperative rows not raised, unchanged) gets target = the same teacher OOF q and weight 0.5 x
    its v3 weight. Cells changed: synovitis 3,700, lat_oa 1,582, pf_oa 985; mean target 0.341 -> 0.371,
    0.125 -> 0.155, 0.258 -> 0.342; mean weight ~0.49/0.45/0.44 -> half. Every other cell bit-identical to
    v3_pseudo_full (rebuilt byte-for-byte first). Config `configs/main_armc_pseudo_full_silentq.yaml` =
    main_armc_pseudo_full.yaml with only run_id and targets_version changed; main tree train.py.
  - *Judge:* `qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_silentq_s{}`, 3-seed (2026/2126/2226)
    mean vs Cfull s2026/2126/2226 on gold-58, paired bootstrap 2000. **Accept iff delta >= +0.004 with P >= 0.95,
    else reject.** Per-label deltas reported, not acted on. Output `qa/silentq/compare.txt` (screen `sq_judge`).
  - **RESULT 2026-10-04: TIE, not accepted.** 3-seed gold 0.9254 vs 0.9239, delta +0.0014 [-0.0044, +0.0069]
    P=0.68 (seeds +0.0011 / +0.0028 / -0.0002). Synovitis 0.812 -> 0.829 (+0.017) in the hypothesised direction,
    but lat_oa 0.836 -> 0.818 (-0.017) and pf_oa +0.005 - not acted on. Same synovitis direction as the
    pre-pseudo B blend (+0.030); a hint, not evidence, at n=27 synovitis positives.
- **PRE-DECLARED 2026-10-04: GLM definition grade for "mild" report mentions ("glmgrade").** Hypothesis: a "mild"
  report mention is often gold-negative under the host's thresholds; grading each report against the competition's
  positive definition (internal GLM-5.3-Flash, frozen prompt `qa/glm_text/run.py` md5 81b0f9ea) separates the
  positive from the sub-threshold mild mentions better than one calibrated "affirmed_mild" value. Text pilot on the
  gold-58 reports (fixed priors, nothing fitted): label-source macro +0.0032 [+0.0007, +0.0072] P=0.995, 4 weak
  labels +0.0086 (qa/glm_text/report.md).
  - *Change:* grades over all 4,349 non-gold reports (`qa/glmgrade/run_full.py` ->
    `data/labels/glm_grade_v1.parquet`); targets `data/targets/v3_pseudo_full_glmgrade` (builder
    `qa/glmgrade/make_targets_glmgrade.py`): on med_men, lat_men, med_oa, lat_oa, pf_oa and synovitis every voter
    cell "affirmed_mild" takes the calibrated value of the study's grade (rows whose grade parsed). Grade values
    calibrated like the existing cells on all 58 gold: Beta strength 8 around yes 0.90 / borderline 0.50 / no 0.05 /
    not_mentioned = the label's absent prior, then PAVA yes >= borderline >= not_mentioned >= no. Weights, teacher,
    the 0.5 mix and the silent-weight rule exactly as v3_pseudo_full (rebuilt byte-for-byte first; every other cell
    verified bit-identical). Config `configs/main_armc_pseudo_full_glmgrade.yaml` = main_armc_pseudo_full.yaml with
    only run_id and targets_version changed; main tree train.py.
  - *Judge:* `qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_glmgrade_s{}`, 3-seed (2026/2126/2226)
    mean vs Cfull s2026/2126/2226 on gold-58, paired bootstrap 2000. **Accept iff delta >= +0.004 with P >= 0.95,
    else reject.** Per-label deltas reported, not acted on. Output `qa/glmgrade/compare.txt` (screen `gg_judge`).
  - *Caveat:* mild selection effect - the same gold-58 chose this idea (text pilot) and calibrates the grade cells,
    and it judges the result. Expected model-level effect ~+0.001-0.003, likely below the gate.
  - **RESULT 2026-10-04: TIE, not accepted.** 3-seed gold 0.9251 vs 0.9239, delta +0.0012 [-0.0052, +0.0072]
    P=0.65; every single seed was below its base twin (-0.0016 / -0.0033 / -0.0013). The labels it targeted moved
    the wrong way: lat_oa -0.014, lat_men -0.011 (pf_oa +0.005, med_oa +0.005, med_men +0.007). The label-level gain
    of the text pilot (+0.0032, P=0.995) did not carry through training - the 0.5 teacher mix and the gold
    calibration of the mild cells already absorbed it.
- **2026-10-04: silentq restricted to synovitis ("silentq_syn"), decided by an LB probe, user-approved.** Targets
  `data/targets/v3_pseudo_full_silentq_syn` (builder `qa/silentq_syn/make_targets_silentq_syn.py`): the 3,700
  report-silent synovitis cells get target = teacher q, weight = 0.5 x v3 weight; the synovitis columns equal
  silentq's exactly and every other column equals v3_pseudo_full exactly (both checked). Config
  `configs/main_armc_pseudo_full_silentq_syn.yaml`; seeds 2026/2126/2226 full-data. **This variant was chosen after
  seeing silentq's per-label gold deltas (synovitis +0.017, lat_oa -0.017), so gold-58 is not a fair judge here**:
  `qa/silentq_syn/compare.txt` is information only. Decision: one LB probe; ship as a slot-1 candidate only if
  LB >= 0.948.
- **PRE-DECLARED 2026-10-04: domain-adversarial training against the hospital ("dann"), user-approved.** Branch
  `dann` de0908c (worktree `.worktrees/dann`, from dc4ec67 = the code that trained the base seeds), config
  `configs/main_armc_pseudo_full_dann.yaml` = main_armc_pseudo_full.yaml + `dann_lambda_max: 0.1`,
  `dann_min_site_studies: 10`; targets v3_pseudo_full unchanged.
  - *Hypothesis:* features that cannot tell the hospital apart generalise better to the test studies, so the
    3-seed gold-58 mean improves.
  - *Change:* a site classifier (Linear(d_fused,256)-GELU-Dropout 0.1-Linear(256,n_sites)) on the fusion input
    (5 filled view embeddings + presence bits) behind a gradient-reversal layer; coefficient
    lambda(p) = 0.1 * (2/(1+exp(-10p)) - 1), p = training progress (Ganin & Lempitsky 2015), lambda_max fixed a
    priori and not tuned; site cross-entropy added with weight 1; no site loss on mixup steps. Site = folds_v1
    `site_key`; sites with < 10 training studies share one "other" class (full-data: 67 sites + other, majority
    rate 0.06). The site head is training-only: bundles, checkpoints and inference are unchanged, and
    `dann_lambda_max: 0` reproduces the default run byte for byte (tests/test_dann.py; full suite green).
  - *Context (no gold numbers):* the 3 public test studies' scanners all occur in train (Siemens Avanto fit 262
    studies, Toshiba Vantage 129, GE SIGNA Artist 208), so a large scanner shift is not visible in what we can
    see of the test set; site separability of the shipped features is recorded in qa/dann/site_separability.json.
  - *Judge:* `qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_dann_s{}`, 3-seed (2026/2126/2226) mean
    vs Cfull s2026/2126/2226 on gold-58, paired bootstrap 2000. **Accept iff delta >= +0.004 with P >= 0.95, else
    reject.** Per-label deltas reported, not acted on. Per-epoch site accuracy logged (qa/dann/site_acc.txt).
    Output `qa/dann/compare.txt` (screen `dn_judge`); workers `dn_gpu0`/`dn_gpu1` start after
    `qa/silentq_syn/DONE`, smoke (40 studies, 1 epoch) first.
  - **RESULT 2026-10-05: REJECTED - DANN HURTS.** 3-seed gold 0.9113 vs 0.9239, delta -0.0126 [-0.0252, -0.0010]
    P=0.02; every seed about -0.013. The site head did its job (site accuracy 0.17 -> ~0.07-0.09 by epoch 1, near
    the 6 % majority class), and removing site information cost lat_oa -0.058, mcl -0.068, lat_men -0.043 (synovitis
    +0.036, not acted on). Site/case-mix carries real signal here, and the public test scanners all occur in train.
- **PRE-DECLARED 2026-10-04: stale-teacher test (external review, Q1 follow-up).** Trigger: the CPU check
  `qa/q1_check/` (pre-set rule) found the GLM regrade improves the MIXED targets on gold-58 (6 graded labels
  +0.0042, P=0.91), so the stale teacher half can matter. Hypothesis: the teacher (trained on old labels) holds
  the regraded cells near their old values and cancels the corrected labels.
  - *Arms* (2x2 on the regraded cells = the 3,894 cells whose label target differs between v3_pseudo_full and
    v3_pseudo_full_glmgrade: med_men 724, lat_men 563, med_oa 782, lat_oa 611, pf_oa 884, synovitis 330): base =
    old labels + teacher; glmgrade = new labels + teacher (both trained); **OT** = old labels, teacher removed;
    **NT** = new labels, teacher removed. Removing the teacher drops only its BCE term: w x BCE(z, 0.5r + 0.5q) =
    0.5w x BCE(z, r) + 0.5w x BCE(z, q), so those cells get target r, weight 0.5w; every other cell is
    bit-identical to the parent table (both parents rebuilt byte-for-byte first). Targets
    `data/targets/v3_pseudo_full_q1{ot,nt}` (builder `qa/q1test/make_targets_q1.py`), configs
    `configs/main_armc_pseudo_full_q1{ot,nt}.yaml` (only run_id + targets_version differ).
  - *Deploy gate:* NT vs base, `qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_q1nt_s{}`, 3-seed
    (2026/2126/2226) paired gold-58. **Accept iff delta >= +0.004 with P >= 0.95, else reject.**
  - *Diagnostic, not a deploy gate* (`qa/q1test/interaction.py`): interaction (NT - OT) - (glmgrade - base) with a
    paired bootstrap CI - positive means removing the stale teacher unlocks the corrected labels; also NT -
    glmgrade and OT - base. If NT fails the gate, the full teacher refresh is not pursued, whatever the
    interaction says. Per-label deltas reported, not acted on.
  - *Caveat:* the same gold-58 motivated the regrade, calibrates its grades, passed the Q1 check and judges
    this - a selection effect throughout.
  - *Queue:* 6 runs (OT, NT x 3 seeds) on screens `q1_gpu0`/`q1_gpu1` (`qa/q1test/worker.sh`, shared claims,
    >= 18 GB free), starting after the DANN judge screen ends; `q1_judge` writes `qa/q1test/compare.txt`,
    `interaction.txt`, then `DONE`.
  - **RESULT 2026-10-05: TIE - stale-teacher hypothesis NOT supported.** Gate NT vs base -0.0005 [-0.0069,
    +0.0056] P=0.44 (3-seed means: base 0.9239, glmgrade 0.9251, OT 0.9259, NT 0.9234). Interaction
    (NT-OT)-(glmgrade-base) = -0.0037 [-0.0111, +0.0028] P=0.14: removing the teacher on the regraded cells does
    NOT unlock the corrected labels (if anything the opposite). No teacher refresh; label-side work closed.
- **PRE-DECLARED 2026-10-05: synovitis co-occurrence + second-stage synovitis classifier** (CPU only, plan in
  `qa/synov/PLAN.md`, written before any gold number). One logistic regression (L2, C=1.0) of synovitis on the
  standardised logits of all 12 Cp 5-fold OOF predictions, trained on non-gold rows whose primary report voter
  states synovitis affirmed/negated, applied to main_v5_C's gold predictions. Deploy only if the 12-label macro
  gold-58 delta >= +0.004 with P >= 0.95 (about +0.048 on synovitis alone); otherwise insight only.
  - **RESULT 2026-10-05: not deployed (insight only).** Gold-58: synovitis co-occurs with effusion (phi +0.40),
    lateral OA (+0.34), PF OA (+0.30); others CIs span 0. Our model's synovitis already tracks these (Spearman
    +0.6-0.9). Classifier: synovitis AUC 0.808 -> 0.828 (+0.020, P=0.81), macro +0.0017 [-0.0021, +0.0057] P=0.81;
    rank average +0.0014 P=0.89. Below the +0.004 / P>=0.95 rule. `qa/synov/report.md`.
- **Final candidates on the hardened notebook, 2026-10-05:** slot 1 = 56854249 (main_v5_C + silentq synovitis,
  hardened, LB 0.948); slot 2 = 56857914 (main_v5_C, kernel `rsna-kad-main-v5-c` v2 = hardened notebook md5
  b328ad1a, submitted 17:20 UTC) = public LB 0.948. Both must be selected in the Kaggle UI (user action).
- **PRE-DECLARED 2026-10-05: heavier teacher share in the pseudo-label mix ("t70"), user-approved.** Source: forum
  topic 735304 (Archit Konde: 50/50 label/OOF jumped, "going heavier on the model added a bit more"); we fixed 0.5
  a priori and never tested beyond it. Hypothesis: the image teacher is more reliable than the calibrated report
  label on average, so a larger teacher share gives cleaner targets.
  - *Change:* targets `data/targets/v3_pseudo_full_t70` (builder `qa/t70/make_targets_t70.py`): the exact
    v3_pseudo_full path with `mix_with_teacher(..., mix=0.7)`, i.e. target = 0.3 x label + 0.7 x teacher on every
    mixed cell; same labels, calibration, teacher and weights (silent-cell weight 0.5). v3 rebuilt byte-for-byte
    first; every weight column bit-identical to v3. 0.7 is the single value, fixed before training - no sweep.
    Config `configs/main_armc_pseudo_full_t70.yaml` = main_armc_pseudo_full.yaml with only run_id and
    targets_version changed.
  - *Judge:* `qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_t70_s{}`, 3-seed (2026/2126/2226) mean
    vs Cfull s2026/2126/2226 on gold-58, paired bootstrap 2000. **Accept iff delta >= +0.004 with P >= 0.95, else
    reject.** Per-label deltas reported, not acted on. Output `qa/t70/compare.txt` + `qa/t70/DONE` (screen
    `t70_judge`); workers `t70_gpu0`/`t70_gpu1` share a claim queue (`qa/t70/worker.sh`, >= 18 GB free).
  - **RESULT 2026-10-06: TIE, not accepted.** 3-seed gold 0.9233 vs 0.9239, delta -0.0007 [-0.0074, +0.0058]
    P=0.44 (seeds +0.0002 / -0.0012 / +0.0006). lat_men -0.025, lat_oa -0.017, fracture +0.021, med_men +0.011
    (not acted on). The 0.5 mix is not improved by leaning harder on our (weaker) teacher.
- **PRE-DECLARED 2026-10-06: multi-source teachers ("mt", forum idea, topic 735304), user-approved.** Hypothesis:
  the v3 teacher (mean of arms A, E, C, all trained on the same Qwen labels; gold ~0.904) shares one label
  source's errors; teachers trained on different weak-label sources err differently, so their mean is a better
  teacher and v3-style students trained on it improve.
  - *Sources* (one voter each, same targets_v1 calibration/weight recipe, fold-local calibration): Qwen =
    `data/labels/llm_v1.parquet` (targets v1; teacher reused: `runs/main_armc_convnextv2_t_384`, the same arm-C
    recipe and seed); public table = `data/labels/src_public.parquet` (merged_v1_1's `_v3` voter as the only
    voter, built by `qa/mt/make_labels_src.py`; no evidence check, parse_ok True) -> `data/targets/v1_pub`;
    Gemma-3-27B = `data/labels/llm_v1_gemma3.parquet` -> `data/targets/v1_gemma`. Self-check: the same
    `scripts/make_targets.py` path rebuilds data/targets/v1 bit-for-bit. GLM grades (6 labels only) not used.
  - *Teachers:* `configs/main_armc_convnextv2_t_384_{pub,gemma}.yaml` = the v1 arm-C teacher config with only
    run_id and targets_version changed; 5 folds each (10 runs). New teacher = equal-weight mean of the three
    source teachers' OOF. Teacher gold-58 AUCs (`qa/mt/teacher_report.txt`) are information only - no gate
    between teachers and students.
  - *Students:* `data/targets/v3_pseudo_full_mt` = v3_pseudo_full with ONLY the teacher replaced (same labels,
    calibration, 0.5 mix, weights; builder `qa/mt/make_targets_mt.py` rebuilds v3_pseudo_full byte-for-byte
    first and asserts weights unchanged); `configs/main_armc_pseudo_full_mt.yaml` (run_id + targets_version
    only); full-data seeds 2026/2126/2226.
  - *Judge:* `qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_mt_s{}`, 3-seed mean vs Cfull
    s2026/2126/2226 on gold-58. **Accept iff delta >= +0.004 with P >= 0.95, else reject.** Per-label deltas
    reported, not acted on. Selection-effect caveat as for every 10-0x test (same gold-58 throughout).
  - *Queue:* screens `mt_gpu0`/`mt_gpu1` (`qa/mt/worker.sh`, atomic claims, >= 18 GB free): 10 teacher folds,
    then `qa/mt/build.sh` (assemble OOF, teacher report, student targets, student smoke) -> `qa/mt/TARGETS_OK`,
    then 3 student seeds; `mt_judge` writes `qa/mt/compare.txt` + `DONE`.
  - **RESULT 2026-10-07: TIE, not accepted.** 3-seed gold 0.9252 vs 0.9239, delta +0.0013 [-0.0056, +0.0080]
    P=0.65 (seeds +0.0034 / +0.0000 / +0.0014). Per label: contusion +0.018, fracture +0.014, med_men +0.011;
    lat_men -0.015, baker -0.011, effusion -0.009. Follow-up 2026-10-10: using mt only on the labels it improved
    is selection on the test set - chosen on 29 studies and scored on the other 29 (2000 splits) it gives
    +0.0015 P=0.74, the same as averaging all 6 models (+0.0014 P=0.74). Not deployed.
- **main_v5_C + public 2.5D ConvNeXt reader, rank blend 0.75/0.25 - submitted 2026-10-10 (57022888), user-approved
  probe.** Reader = `goodpjw2008/rsna-knee-2-5d-convnext-reader` (3 folds; the notebook author's own model inside
  public notebook `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`; reader alone LB 0.929). Its
  make_labels keeps gold at fold -1 and train.py saves the last-epoch EMA, so gold-58 is a clean holdout for it.
  Weight 0.25 fixed before scoring. Gold-58: main_v5_C 0.9255, reader 0.9105, blend 0.9282, +0.0027 [-0.0031,
  +0.0089] P=0.82 (below the rule; probe only); per-label Spearman ours vs reader 0.77-0.95. Kernel
  `rsna-kad-v5c-reader` v2 (`qa/reader/build_kernel.py`): hardened notebook b328ad1a run unchanged as leg 1, the
  reader as leg 2 only with >= 1.5 h left; any reader failure keeps leg 1's file (tested with the reader missing).
  Kaggle output = local bit for bit. v1 ran on Kaggle's new default image (torch 2.11), where our decoder wheels
  do NOT install; v2 pins the image main_v5_C scored on (sha256:37c64f7d..., T4). Every future kernel must pin it.
  Result: LB 0.948 = tie with main_v5_C.
- **Public OAI stack 0.7 + main_v5_C 0.3, rank blend - submitted 2026-10-10 (57037844), user-approved probe.**
  Stack = public notebook `rabari9999/rsna-knee-sota-0-954`, its own code unchanged: nartaa CoAtNet-rmlp-2 384
  SWA (LB 0.949 alone) and 224-crop SWA (0.945) from `nartaa/rsna-knee-publication-swa-weights-20261007`, the
  goodpjw2008 reader, hand-set per-finding reader weights. nartaa's README: both checkpoints used 2,399 OAI knees
  and were selected on gold-58. OAI status: host 741819 (09-21) allows OAI only "without an escalated (i.e.
  institutional) review and sign-off", and NDA lists OAI as "Institutional Approval Required: No", but access is
  blocked from China; the host has not answered (as of 10-10) whether OAI-trained public weights are allowed.
  Plan: one final slot for this, one for clean 56857914. W = 0.3 fixed before scoring. Gold-58 (qa/oaistack/
  gold_score.py): c384 0.9229, c224 0.9149, reader 0.9153, stack 0.9302, main_v5_C 0.9255, blend 0.9342;
  blend - stack +0.0040 [-0.0015, +0.0099] P=0.93, blend - ours +0.0088 P=0.95; Spearman stack vs ours 0.85-0.94.
  The stack's weights were fitted on gold-58 (its per-label table matches our gold numbers), so the stack's score is
  optimistic and the delta understates our leg. Kernel `rsna-kad-v5c-oaistack` v1 (qa/oaistack/build_kernel.py,
  blend md5 66b2ccf7): hardened b328ad1a as leg 1, the stack as leg 2 only with >= 1 h left; any stack failure
  keeps leg 1's file. E2E on gold-58 (qa/oaistack/e2e.sh): blend 0.934, reader-missing fallback == leg 1 byte for
  byte. A6000 timing on 58 studies: ours 116 s, stack 119 s, so the hidden run should be ~2x main_v5_C (~3 h).
  Result: LB 0.953 (best so far, +0.005 over main_v5_C). The stack alone was not submitted by us (its notebook claims 0.954).
- **stk: main_v5_C students re-taught by the public OAI stack - started 2026-10-10 21:06, user-approved.**
  Teacher = the stack (nartaa c384/c224 + goodpjw reader, fusion as in the oaistack kernel) run on all 4,407
  train studies (qa/stk/teacher_run.sh, fake /kaggle tree). Its ranks are quantile-mapped onto the old v3 teacher's
  values per finding (marginals unchanged; only the order changes), then v3_pseudo_full is rebuilt with only the
  teacher replaced -> data/targets/v3_pseudo_full_stk (qa/stk/make_targets_stk.py; dry run on random teachers
  rebuilt v3 byte for byte). Students: configs/main_armc_pseudo_full_stk.yaml = main_v5_C's recipe, seeds
  2026-2526. Caveat: nartaa trained on these studies, so the teacher is partly in-sample.
  - *Queue:* screens stk_g1a (lead: teacher -> targets -> smoke -> students), stk_g1b, stk_g0 (qa/stk/worker.sh,
    atomic seed claims at >= 18 GB free); the last worker writes qa/stk/compare.txt + DONE (qa/stk/judge.py).
  - *Pre-declared gate:* gold-58 blend(stack 0.7, stk_C 0.3) vs blend(stack 0.7, main_v5_C 0.3), i.e. a
    replacement for slot 1 (57037844, LB 0.953). Delta > 0 -> one LB probe (oaistack kernel with stk_C weights);
    adoption is decided on that probe. stk_C alone vs main_v5_C is information only.
  - *Result 2026-10-11 (qa/stk/compare.txt): STOP, nothing submitted.* gold-58: blend_new 0.9338 vs blend_old
    0.9342, delta -0.0004 [-0.0041, +0.0032] P=0.40. stk_C alone beats main_v5_C (0.9323 vs 0.9255, +0.0069
    P=0.95; single seeds 0.920-0.935) but moved toward the stack (rho with stack +0.90..0.97 vs +0.85..0.94),
    so the diversity it adds to the blend is gone. Slot 1 stays 57037844 (LB 0.953).
  - *LB probe anyway, user-requested 2026-10-11 (overrides the gate):* submission 57070607 = kernel
    sroziews/rsna-kad-v5c-stk-oaistack v1 (blend script unchanged, md5 66b2ccf7) on dataset
    sroziews/rsna-kad-weights-v5c-stk (export/main_v5_C_stk, qa/stk/export_stk.py). Local e2e on gold-58 0.9338
    (qa/stk/e2e_stk.sh); Kaggle log: 6 stk members loaded, blended. LB: 0.952 (tie with 57037844's 0.953).
- **v2stack probe, user-requested 2026-10-11 (public-notebook scan, option 1).** The 10-10/10-11 public frontier
  is one recipe: matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed = nartaa Raptor CoAtNet 384 SWA (OAI) 0.7 +
  heliosli compact ConvNeXt-small S6 (bundle 209a32c8b50a6b1c, 4,344 competition studies, no OAI) 0.3, claimed
  0.954; velci-kcd25-20 adds a private member at 0.2 (claimed 0.955). Scan artefacts: server
  qa/discussion_2026-10-11/ (forum crawl5.py, pulled kernels/). Forum 735304 reports non-OAI single 2.5D
  ConvNeXt-T at 0.953-0.959 public, so our 0.948 single is below the field, not at a ceiling.
  - *Kernel:* qa/v2stack/build_kernel.py: leg 1 = hardened main_v5_C (md5 b328ad1a, fallback file); leg 2 = that
    notebook's cells verbatim (sha256 d358d045) but one line, NB_T0 = the wrapper start (its 8.3 h deadline would
    otherwise ignore our leg's time); final = 0.7 rank(base) + 0.3 rank(main_v5_C), W fixed before any gold number.
  - *e2e on gold-58 (qa/v2stack/e2e.sh):* full run blends 58/58 (Raptor 0 failed, compact 0 failed); with the
    heliosli bundle hidden leg 2 fails and submission == leg 1. Server-only test quirks handled in e2e.sh, not in
    the kernel: distutils .pth, and the heliosli controller sets CUDA_VISIBLE_DEVICES=<worker> itself (one worker
    per nvidia-smi GPU), so a shim reports one GPU and a python wrapper maps it to the pinned one.
  - *gold-58 (qa/v2stack/gold_score.py):* Raptor 0.9229, compact 0.9263, base 0.9313, main_v5_C 0.9253, blend 0.9338;
    blend - base +0.0026 [-0.0029, +0.0084] P=0.80; rho(base, ours) 0.90. Same level as 57037844 (0.9342): an LB tie
    is the expected outcome. Base gold is optimistic (Raptor selected on gold-58).
  - *Submission 57084313* = kernel sroziews/rsna-kad-v5c-v2stack v1 (blend md5 bcef1b1e); Kaggle log on the 3
    public studies: both legs ran, blended, 224 s. **LB 0.955** (2026-10-12) - new best, +0.002 over 57037844
    (within LB noise ~0.02, but at least a tie). Final slot 1 = 57084313; slot 2 = 56857914 (clean main_v5_C).
  - *2026-10-12 checks:* heliosli's Raptor 0.5 + compact 0.5 notebook uses the same two models (same bundle
    209a32c8b50a6b1c, Raptor = rabari's c384, rho 1.000); gold blend 0.9352 vs 0.9338, rho 0.992 with 57084313 - not
    probed. Every remix of the public parts (stack + v2 base, + compact into 57037844) stays within +0.003 on gold.
- **OAI training (qa/oai, user-approved 2026-10-12).** Hypothesis: MOAKS reads of 2,439 OAI baseline knees add real
  supervision on our three weakest findings (PF OA, Lateral OA, Synovitis), as they did for nartaa's Raptor.
  - *Data:* NDA package 1250340 (user's account; shared OAIBaselineImages), image03.txt maps barcode -> sequence;
    only SAG_IW_TSE, COR_IW_TSE, COR_MPR, AX_MPR of MOAKS knees downloaded (9,799 series, 75 GB, gzip-verified) to
    /raid/sroziewski/data/oai. MOAKS grades from 3DReasonKnee's CC BY re-export (header shifted by one column).
  - *Layout:* qa/oai/prep_dicom.py -> competition layout (study oai_<ID>_<R|L>; SAG_IW_TSE = sag_fs, MPRs = cor_fs /
    ax_fs, COR_IW_TSE = cor_t1; no sag_t1). OAI MPR series start with one off-orientation image that collapsed the
    stack to one slice - only the majority orientation is kept. Cache built with cache_v1 params into
    cache/rsna_kad/oai_v1; merged root cache/rsna_kad_oai/v1 symlinks both (version string stays v1).
  - *Labels (fixed before training, qa/oai/make_inputs.py):* PF/Lateral OA = 1 if full-thickness loss or area grade
    3, 0.5 (w 0.5) if area grade 2, else 0; Synovitis = Hoffa 2-3 -> 1, 1 -> 0.5 (w 0.5), 0 -> 0; all other findings
    weight 0. Targets v3_pseudo_full_oai = v3_pseudo_full + OAI rows.
  - *Run:* configs/main_armc_pseudo_full_oai.yaml = main_v5_C's config but targets/folds; 6 seeds (s2026..s2526),
    qa/oai/worker.sh on GPUs 1 and 3.
  - *Gate (pre-declared, qa/oai/judge.py):* blend(v2 base 0.7, OAI C 0.3) vs blend(v2 base 0.7, main_v5_C 0.3) on
    gold-58 (self-test with old = new reproduces 0.9338). Delta > 0 -> one LB probe; adoption on that probe. Risk
    (stk lesson): OAI pulls our leg toward Raptor, so rho(base, OAI C) is reported.
  - *Result 2026-10-13 (qa/oai/compare.txt, 6 seeds done 00:01): gate passes by a hair -> one LB probe.* gold-58:
    blend_new 0.9343 vs blend_old 0.9338, delta +0.0005 [-0.0030, +0.0038] P=0.61. OAI C alone is worse than
    main_v5_C (0.9206 vs 0.9255, -0.0048 P=0.16; single seeds 0.910-0.924); the MOAKS-mapped findings did not
    improve alone (Lateral OA 0.816 vs 0.832, PF OA 0.879 vs 0.884; Synovitis 0.838 vs 0.808 did). rho with the base
    unchanged (+0.83..0.93), so no stk-style collapse. Expected LB gain is below resolution.
  - *Probe build:* export/main_v5_C_oai (qa/oai/export_oai.py: main_v5_C's ensemble.json with the 6 OAI members)
    -> private dataset sroziews/rsna-kad-weights-v5c-oai; kernel sroziews/rsna-kad-v5c-oai-v2stack =
    qa/v2stack/build_kernel.py --weights-dataset/--kernel-id (new options; defaults rebuild 57084313's kernel byte
    for byte, blend md5 bcef1b1e unchanged, only the leg-1 dataset differs). e2e on gold-58: qa/v2stack/e2e.sh with
    OURS_W=export/main_v5_C_oai TAG=oai_, scored by qa/oai/e2e_score.py (expected submission 0.9343).
  - *e2e 2026-10-13:* leg 1 0.9206, base 0.9313, submission 0.9343 on gold-58 - exactly judge.py's numbers; 58/58
    blended, 0 failures, 457 s. Kaggle kernel v1 on the 3 public studies: 6 OAI members loaded, blended, 223 s.
  - *Submission 57134537* (2026-10-13 07:10, kernel sroziews/rsna-kad-v5c-oai-v2stack v1). Adopt for slot 1 only if
    it beats 57084313's 0.955; a tie keeps 57084313 (fewer moving parts, no OAI-trained leg of our own).
  - *LB 2026-10-13: 57134537 = 0.955, a tie with 57084313 -> not adopted; slot 1 stays 57084313.* OAI line closed.
- **v3stack: R7 base + main_v5_C, user-approved 2026-10-13 (public-notebook scan).** Every public 0.956-0.957 notebook
  is heliosli/0-957-latest-dataset-score-in-one-click (sha256 cb2a0f57..., pulled to qa/v3stack/src) or a copy of it:
  nartaa Raptor 384 0.5 + heliosli compact R7 ConvNeXt-small (dataset rsna-knee-compact-r7cnxfsss, bundle
  8c7b666e27dc0e8f, 4 folds, crop 352, fastMRI aux, refitted stackers) 0.5. Variant varshithakolupuri adds nartaa's
  c224 Raptor at 35 % of the Raptor half (same displayed 0.956). Pulled kernels: qa/discussion_2026-10-13/kernels.
  - *Kernel:* qa/v2stack/build_kernel.py gained --nb/--nb-sha/--base-datasets (code cells only; defaults rebuild the
    v2stack and OAI kernels byte for byte). v3stack = the R7 notebook as leg 2, main_v5_C leg 1, W 0.3 as in 57084313,
    image pinned to 37c64f7d (main_v5_C's). Driver qa/v3stack/e2e_v3.sh; verdict qa/v3stack/gold_score.py.
  - *Gate (pre-declared):* blend > R7 base on gold-58 -> LB probe; plain R7 is submitted regardless as a baseline.
  - *Gold-58 (e2e, 58/58 blended, 0 failures):* R7 base 0.9356, main_v5_C 0.9253, v3 blend 0.9403; blend - R7
    +0.0047 [-0.0007, +0.0101] P=0.95; R7 - v2 base +0.0043 P=0.91; v3 blend - 57084313's blend +0.0065 [+0.0014,
    +0.0128] P=0.99. Gains on MCL, Synovitis, Fracture, Effusion. Bias: R7 trained with gold in its pool -> against us.
  - *Submissions:* 57150688 = plain R7 (kernel sroziews/rsna-kad-r7-plain v1, heliosli's notebook and image
    unchanged); 57150980 = v3stack (kernel sroziews/rsna-kad-v5c-v3stack v1; Kaggle log on the 3 public studies:
    6 members loaded, 3/3 blended, 0 failures, 542 s). Adopt v3stack for slot 1 if it scores >= plain R7.
  - *LB 2026-10-13 23:30: v3stack 57150980 = 0.957 (new best), plain R7 57150688 = 0.956.* Slot 1 -> 57150980.
    Gold predicted the order (0.9403 > 0.9356 > 0.9338) and LB agrees (0.957 > 0.956 > 0.955).
  - *c224 check, user-requested 2026-10-13 (qa/v3stack/c224_gold.py -> c224_compare.txt): STOP, nothing submitted.*
    Raptor slot = 0.65 c384 + 0.35 c224 (varshithakolupuri's A224) inside v3stack, both arms rebuilt from the v3 e2e
    legs (rebuilt v3 0.9405 vs kernel 0.9403; oaistack c384 == v3 run's Raptor, rho 1.000): v3_c224 0.9401 vs v3
    0.9405, -0.0005 [-0.0040, +0.0026] P=0.41, despite a gold bias in c224's favour (nartaa selected it on gold-58).
    c224 alone 0.9149, rho with c384 0.944; it helps Lateral/Medial OA and Effusion, costs Baker's, Synovitis, PF OA.
- **Forum scan 2026-10-14 (qa/discussion_2026-10-14: crawl6.py, new_since_1011.txt).** One new topic (748295) and 64
  new comments since 10-11. Ideas: train only on cells LLM labellers agree on (745861, Prateek Grover: -25 % of
  cells, "improved CV and public LB"; hengck23: let the LLM say "don't know"); top teams say "labels ... AI, not by
  hand, and don't rely on reports alone" (azamat1ch, ~0.964); random / other-body-part negatives (hengck23). Open
  risk: posts of 10-13 still ask the host to rule on OAI-trained public weights (nartaa, heliosli orthofoundation)
  for LB/medal and prize eligibility - both 57150980 and 57150688 contain Raptor; 56857914 is the OAI-free hedge.
- **agree: main_v5_C students with uncertain label cells masked, user-approved 2026-10-14.**
  - *Targets:* data/targets/v3_pseudo_full_agree (qa/agree/make_targets_agree.py) = v3_pseudo_full with weight 0
    where the voters (llm_v1, src_public, llm_v1_gemma3, glm_grade_v1 on its six findings) contradict each other or
    fewer than two give a definite call while one speaks; 11.8 % of weighted cells (contusion 1,072, lat_oa 823,
    fracture 809 ... effusion 121); targets otherwise identical. Basis, measured before training: on gold-58 llm_v1's
    definite calls are wrong in 8/17 contested and 6/15 weak cells vs 96/467 elsewhere.
  - *Run:* configs/main_armc_pseudo_full_agree.yaml = main_v5_C's config but run_id/targets_version; seeds
    s2026..s2526; qa/agree/worker.sh screens agr_w1 (lead), agr_w1b on GPU 1, agr_w3 on GPU 3; smoke passed.
  - *Gate (pre-declared, qa/agree/judge.py):* blend(R7 base 0.7, agree C 0.3) vs blend(R7 base 0.7, main_v5_C 0.3)
    on gold-58 (base = v3 e2e's _base.csv); delta > 0 -> one LB probe (v3stack kernel with an agree weights dataset).
  - *Result (6 seeds, done by 2026-10-14 09:30, qa/agree/compare.txt):* **STOP, nothing submitted.** blend_new
    0.9391 vs blend_old 0.9403, -0.0011 [-0.0037, +0.0012] P=0.17; agree C alone 0.9245 vs main_v5_C 0.9255
    (-0.0010, P=0.36; single seeds 0.917-0.922). Masking helped ACL/MCL/Medial OA/Effusion and hurt Lateral OA
    (-0.020), Lateral Meniscus, PF OA - the masked cells concentrate in the hard findings, where they were also the
    signal. Agreement masking closed; main_v5_C stays the C leg.
- **Scan 2026-10-14 evening (qa/discussion_2026-10-15: crawl7.py, new_since_1014.txt, kernels/).**
  - *Host ruling (743416, Chris Carr, RSNA, 10-14):* no dataset list; external data is fine if it needs only
    registration + licence acknowledgement, not if it needs an institutional agreement/affiliation; disclose datasets.
    OAI via NDA was obtained here with a personal account only, so the OAI-weights risk on slots 1/2 is lower; the
    geographic-block follow-up (Driz Lo) is unanswered.
  - *Code:* nothing new beyond the R7 family. heliosli's "=0.9573" = R7 0.5 + Raptor half (0.65 r384 + 0.35 c224),
    i.e. the c224 variant our gold check rejected (qa/v3stack/c224_compare.txt). bliverrigpdoge 0.957 = R7 notebook
    0.8 + private knee-centred ConvNeXt-s/EffNetV2-S 0.2 (same pattern as our v3stack). goodpjw2008 2.5D ConvNeXt
    reader at full weight costs (0.956 -> 0.954), kept at 1/10; S6 as 0.3 of the compact slot; 224 mirror = same
    0.956. rabari9999 "micro-boost" = probit/per-finding hand weights, no new model. No public notebook above 0.957.
- **v4pf: per-finding weight for our leg, user-approved 2026-10-14 as a long shot.** v3stack with main_v5_C at 0.5
  instead of 0.3 on Effusion, Fracture, Baker's (where it beats the R7 base on gold: .989/.961/.976 vs
  .969/.940/.968). qa/v2stack/build_kernel.py gained `--w-per JSON` (default builds of v2stack and v3stack
  byte-identical, checked); kernel sroziews/rsna-kad-v5c-v4pf v1 (qa/v4pf/kernel_blend, e2e qa/v4pf/e2e_v4pf.sh:
  kernel output == offline blend). Gold 0.9415 vs v3stack 0.9403, **circular** (findings picked on the same gold-58),
  so no gate - the LB decides. Kaggle run COMPLETE, submitted 2026-10-14 20:00 as **57193974** (pending).
- **Team merge with Qile Jiang (qilejiang76), accepted 2026-10-14.** His 0.957 = his own models only (1 ResNet-224 +
  2 ConvNeXt-T-384; Opus-5.5 report labels + public mean-of-tables; one leg OAI). Hosted-LLM labels are allowed
  (host, 733965, 2026-08-27). Our LB-vs-gold plot for him: qa/lbgold (plot.py, bundle_gold.py; Spearman 0.95 over 22
  submissions, but gold cannot separate near-ties).
- **Joint kernel ready, partner code pending (qa/joint).** joint_template.py + build_joint.py: leg V = the v3stack
  kernel embedded verbatim except its start time (JOINT_T0, one shared 9 h budget); leg P = partner .py/.ipynb run
  unchanged as its own process (cwd /kaggle/working, its submission.csv is its output; `!`/`%` lines commented out);
  per-finding rank blend `--w-partner` (default 0.5); either leg failing, timing out (`--partner-max-h`) or missing
  a study leaves the other's file; `--partner-first`; kernel metadata carries v3stack's datasets + image,
  `--partner-datasets`, `--docker-image`. e2e qa/joint/e2e_joint.sh with a stand-in partner (OAI-C predictions):
  ok, partner-first, fail, partial, timeout all PASS (check_joint.py; leg V within 3 rank steps of the reference -
  the R7/Raptor base is not bit-reproducible on GPU, one run swapped two Medial Meniscus ranks).
- **Joint kernel with Qile's real model, 2026-10-15 (qa/qile).** Team merged (Szymon Roziewski: sroziews +
  qilejiang76, 67 submissions, shared 5/day). Partner = his notebook qilejiang76/rsna-knee-ours-mtr1-mth4-olp5 v2
  (LB 0.957, submission 57127512: MT-r50 0.1 / MT-hires 0.4 / OL-plane 0.5, 4 x 5-fold ConvNeXt-T/ResNet-50, rank
  blend; datasets qilejiang76/rsna-knee-lineage-g1-mth-mtr-olp-20261013 + rsna-knee-single-models-20261007; his
  image 2757e0c7, py3.13), copied to qa/qile/qile_v2.ipynb. His gold58 predictions (5-fold means, never trained on
  gold) + Opus Blend labels: server .tmp/qile/in.
  - *Gold-58 (qa/qile/gold_joint.py -> gold_joint.txt, information only):* v3stack 0.9403, Qile 0.9399, joint
    0.5/0.5 0.9437 (0.3..0.7 all 0.9415-0.9437), delta vs v3stack +0.0034 [-0.0024, +0.0093] P(<=0)=0.13;
    Spearman v3stack ~ Qile 0.940 (R7 ~ ours 0.887).
  - *e2e with his real code (qa/qile/e2e_qile.sh, check_qile.py): PASS* - his leg ran unchanged in the venv
    standing in for our image, P == his offline gold58 (Spearman 0.9999), blend 0.9440; A6000 gold-58 legs V 464 s,
    P 217 s, so the hidden joint run is ~1.5x v3stack's.
  - *Kaggle kernel sroziews/rsna-kad-joint:* push 1 rejected (HTTP 400: joint.py 1.19 MB; Kaggle caps script size
    near 1 MB) -> embedded legs zlib+base64 (436 KB; decoded legs byte-identical). v1 crashed at 3 s: our pinned
    image mounts the competition at /kaggle/input/<slug>, not /kaggle/input/competitions/<slug> (the fake tree has
    the latter) -> joint_template tries both. v2 COMPLETE on the 3 public studies: V 460 s, P 922 s (~700 s of it
    his recursive /kaggle/input globs walking the competition DICOM tree, 3 of them), blend 3/3; his 4 models'
    fold means equal his own run's output exactly (our image vs his). sroziews/rsna-kad-joint-dbg2 = a stray
    v3stack copy pushed while diagnosing the 400. User-approved: v2 submitted 2026-10-15 09:38 UTC = **57225242**,
    LB pending.
  - *Merge requests 10-15:* balamanikandan (0.956, 2 members, 151 subs) and Nicolai Karcher / pejophylo (0.957
    solo, 112 subs; own models 0.947/0.938/0.938, rest public notebooks). Both asked for gold-58 predictions of
    their own models first; merge only if > +0.003 on gold, runtime fits, Qile agrees.
    Nicolai sent 3 gold-58 files (`.tmp/nic/gold_nic.py/.txt`): singles 0.895-0.918, mean 0.922, Spearman ~ joint
    0.938; joint + his mean at 0.2-0.5 = 0.9403-0.9349 vs joint 0.9437 (0.25: -0.0036 P(<=0)=0.96) -> decline
    recommended. Rifti (rifti86793, radiology resident, solo, 0.942 from 2 subs, rank 2674): nothing to blend,
    decline recommended.
  - *Fast partner globs (2026-10-15):* the v2 log shows Qile's three `glob('/kaggle/input/**/NAME', recursive=True)`
    calls walking the competition DICOM mount took 703 + 153 s of leg P's 922 s on 3 studies. `build_joint.py
    --partner-fast-glob` rewrites them into an os.walk that prunes rsna-knee-abnormality-detection (each NAME is in
    one dataset only; default build still byte-identical to v2). Gold e2e qa/qile/e2e_qile_fast.sh PASS (blend
    0.9440, P == his offline). Kernel sroziews/rsna-kad-joint **v3** (qa/qile/kernel_joint_fast): leg P 922 -> 103 s,
    total 1383 -> 611 s, submission.csv identical to v2's. Not submitted (57225242 = v2 still pending).
  - ***57225242 (joint v2) scored LB 0.959*** (seen 2026-10-15 18:40 UTC; both solo legs 0.957): new team best.
- **PRE-DECLARED 2026-10-15: Opus-label C leg (qa/opus), user-approved.** Pre-check (`qa/opus/label_gold.py/.txt`):
  the label tables alone on gold-58 - Qile's Opus Blend soft labels 0.9364 vs our calibrated v1.1 report targets
  0.8728, +0.064 [+0.044, +0.084], better on all 12 findings (synovitis .815 vs .674, contusion .927 vs .805);
  Spearman between the two on the 4,349 non-gold studies 0.48-0.90. Open question to Qile: was the Opus Blend
  tuned on gold-58 (would flatter every gold number below).
  - *Targets* `data/targets/v3_pseudo_full_opus` (`qa/opus/make_targets_opus.py`; rebuilds v3_pseudo_full
    byte-for-byte first): v3_pseudo_full with only the label part replaced by Opus soft labels; same main_v3
    teacher, 0.5 mix, gold excluded; weight = max(0.1, Opus confidence). `qa/opus/target_stats.csv`.
  - *Runs* `configs/main_armc_pseudo_full_opus.yaml` (run_id + targets_version only), full data, seeds
    2026/2126/2226; screens opus_gpu1 / opus_gpu3 (`qa/opus/worker.sh`), started 2026-10-15 20:50 UTC.
  - *Judge* (screen opus_judge, `qa/opus/judge.sh`): solo = `compare_pw.py` 3-seed mean vs Cfull same seeds
    (`qa/opus/compare.txt`, information); **decision = `qa/opus/gold_joint_opus.py`: joint (0.5 x (0.7 R7 + 0.3 C)
    + 0.5 Qile) with the Opus C seeds vs the old C seeds; rebuild the joint kernel only at delta >= +0.004 with
    P >= 0.95**, else keep the current joint. Self-test with old == new gives +0.0000 (JOINT old C 3 seeds 0.9426).
    Risk named up front: an Opus-trained C leg moves toward Qile's models, so the joint may gain less than the leg.
  - *Andrey Tikhomirov (andreytikhomirov, 0.955 solo, 34 subs; claims own CoAtNet + localization, external data,
    0.948 alone) gold-58 (`.tmp/andrey/gold_andrey.py/.txt`, his latest_gold.csv):* alone 0.9373; Spearman ~ joint
    0.916, ~ Qile 0.892 (v3stack ~ Qile 0.938) - the most independent candidate so far; joint + Andrey 0.10-0.50 =
    0.9447-0.9455 vs 0.9437, best 0.15: +0.0018 [-0.0018, +0.0057] P(>0)=0.83 - positive at every weight but under
    the +0.003 merge bar. **Merged 2026-10-15.** His answers: gold never trained on (all-data models on non-gold
    studies, no gold epoch selection), but per-finding label-source choices and the per-finding mix were picked on
    gold, so 0.937 is somewhat optimistic. Own models only: two CoAtNet-RMLP-2 MIL ("hx" 12 positions, metadata,
    competition + external, public 0.945; "n48" 48 positions, competition only), YOLOX knee localisation; mix per
    finding on ranks (Fracture/Lat Men/Lat OA = n48, Synovitis/Effusion = hx, rest mean) = public 0.948. External:
    MRNet (registration + research licence) and KneeMRI (Zenodo, CC BY-NC-ND 4.0) as pseudo-labelled studies; no
    OAI/fastMRI. Hidden runtime 2xT4: hx 52 min, both 106 min.
    **Joint3 kernel (qa/andrey, 2026-10-15):** joint_template/build_joint gained `--extra NAME:W:SRC[:WORK_ENV:SUBDIR]`
    legs (run after V and P, own process, own timeout `--extra-max-h` 3 h; weights V (1-wP)(1-wX), P wP(1-wX), X wX,
    renormalised over the legs that succeed). A = his hx48_ensemble.py unchanged (HX48_WORK=<work>/hx48), w 0.2 fixed
    before any LB number. Gold e2e `qa/andrey/e2e_joint3.sh` + `check_joint3.py`: ok case legs V 797 s / P 332 s /
    A 155 s, submission == .4V+.4P+.2A exactly (1e-16), gold **0.9462** vs 0.9439 for the V/P blend of the same run;
    afail (A exits 1) PASS, submission == .5V+.5P. The ok case is marked FAIL only on my A-reproduction bar
    (Spearman > 0.99 vs his latest_gold.csv: 0.987, per finding 0.977-0.995): his README says latest_gold.csv is the
    last-epoch variant (0.937) while the scored kernel's checkpoints give 0.936 - our leg gives 0.9365, i.e. it
    reproduces the scored kernel, not that file. Kaggle kernel sroziews/rsna-kad-joint3 v1 (qa/andrey/kernel_joint3,
    pinned image 37c64f7d) on the 3 public studies: V 460 s, P 78 s, A 62 s, 0 failures, blend .4/.4/.2.
    User + Qile approved: submitted 2026-10-15 22:23 UTC = **57245439** (last submission of the day), LB pending.
    His validation findings: gold-58 has no normal knees (~13 % of train has no finding), so it under-rates what
    helps on normal anatomy (his MRNet pseudo-labels: gold -0.009, public +0.003); single-finding LB probes put
    Contusion ~0.04 and ACL ~0.011 lower on LB than gold, Synovitis/PF OA/Lateral OA 0.02-0.045 higher; LB-gold gaps
    of 12 single models spread at noise level except extra-knee models (nartaa +0.026, his MRNet +0.024).
    Moazzem Hossain (Invisible_Ensemble, 0.957, 2 members) and Matvey Murtazaliev (0.956 solo, 21 subs): no model
    described; asked for gold-58 predictions.
  - *Qile's answer (2026-10-15 21:08):* prompts = target definitions, not tuned; confidence = product of experts,
    not tuned; **blend weights tuned on gold-58**. So the 0.9364 label pre-check and every gold number of the Opus
    leg carry a selection bonus (a few weights, not per-study labels; our own calibration also used all 58 gold,
    but its per-state values barely move ranks). **Amendment, declared before any result:** passing the gold gate
    only earns one LB probe; the Opus joint replaces the current joint as a final pick only if that probe is
    >= 0.959.
  - ***Opus C leg verdict (judge 2026-10-16 00:29 UTC): TIE/REJECT, no LB probe.*** Solo 3-seed mean 0.9226 vs
    0.9239 old, -0.0013 [-0.0108, +0.0084] P=0.39 (contusion +0.031, fracture +0.013; lat OA -0.029, lat men -0.027,
    MCL -0.014). Joint opus C 0.9429 vs old C 0.9426, +0.0003 [-0.0024, +0.0029] P(>0)=0.58. The named risk
    happened: Spearman C ~ Qile 0.900 -> 0.915. Failed even with the gold-tuned label bonus, so the C leg stays on
    our labels; closed.
  - ***57245439 (joint3) scored LB 0.959*** (seen 2026-10-16 07:48 UTC), the same as joint v2 at 3 decimals;
    forecast was ~0.960 (+0.001 to +0.002 from the 0.5-0.8x gold->LB mapping of +0.0023). The hidden-run log is
    not available, so a ran-A result cannot be told from the V/P fallback (which would also give 0.959). Andrey's
    own legs on the team list: hx 0.945 (57236870), mixed v3 0.948 (57236838). Recommendation for final picks
    (decision 2026-10-21): joint v2 (57225242) stays the main pick; joint3 is a better second pick than a solo
    0.957 (gold +0.0023, three families) but carries extra runtime and the MRNet/KneeMRI disclosure.
  - *Did leg A run on the hidden test? (user-approved diagnostic, 2026-10-16):* `build_joint.py --require-all-legs`
    makes the kernel exit non-zero with no submission.csv when any leg is dropped (it also deletes P's leftover
    file, which Kaggle would otherwise score). Stub-leg test `qa/joint/test_require_all.py`: 6/6 PASS. joint3 v1 still
    rebuilds byte-for-byte without the flag (the template only gains the REQUIRE_ALL lines). Kernel
    sroziews/rsna-kad-joint3 **v2** (qa/andrey/kernel_joint3_diag) on the 3 public studies: V 467 s / P 85 s /
    A 63 s, submission.csv md5 == v1's. Reading the LB result: **0.959 = all three legs ran on the hidden test, the
    tie with joint v2 is real; error = a leg is dropped there** (most likely A: timeout or input), and v1's 0.959
    was the V/P fallback. Submitted 2026-10-16 08:27 UTC = **57261640: FAILED ("Notebook Threw Exception")**, i.e.
    a leg is dropped on the hidden test and joint3 v1's 0.959 was the V/P fallback; equal-thirds probe cancelled,
    joint3 is no final-pick candidate until fixed. The exception does not say which leg or why (a leg timeout also
    ends in our SystemExit, not in Kaggle's timeout).
  - *Encoded diagnostic (user-approved, 2026-10-16):* `build_joint.py --diag-encode` writes the outcome into the
    score: the first `diag_code` of [ACL, Medial OA, Effusion, MCL, Medial Meniscus, Baker's, Fracture, Contusion]
    are set to 0.5 (about -0.04 macro AUC each). Expected LB: 0 all ok ~0.959 | A timeout with budget >= 2.9 h 1
    ~0.919, 1.5-2.9 h 2 ~0.878, < 1.5 h 3 ~0.837 | 4 A skipped (< 30 min left) ~0.797 | 5 A failed / no file ~0.757
    | 6 A output rejected ~0.718 | 7 P dropped ~0.679 | 8 V dropped ~0.644. Stub test `qa/joint/test_diag_encode.py`
    10/10 (every code on the real code path; buckets on synthetic reports); `test_require_all.py` 6/6. Kernel
    sroziews/rsna-kad-joint3diag v1 (qa/andrey/kernel_joint3diag) public run: code 0, all legs, submission md5 ==
    v1's. Submitted 2026-10-16 17:26 UTC = **57277456** (not a candidate).
  - *Leg weights (gold-58 sweep `.tmp/andrey/weights_grid.txt`):* legs alone V 0.9403 / P 0.9399 / A 0.9365; every
    A weight 0.10-0.50 within 0.001 (0.9454-0.9464); equal 1/3 0.9456 vs current 0.9461 (-0.0005, P=0.38); gold
    optimum 0/.55/.45 (drops V, i.e. favours the two gold-tuned legs) not used. Spearman V~P 0.938, V~A 0.924,
    P~A 0.895. **User-approved plan:** if 57261640 scores 0.959, submit equal thirds (kernel sroziews/rsna-kad-joint3eq
    v1, qa/andrey/kernel_joint3eq, --require-all-legs; public run all legs ok, weights 1/3 each); it becomes the
    joint3 final candidate only at LB >= 0.959, else keep .4/.4/.2.
- **PRE-DECLARED 2026-10-16: C leg + Andrey's external knees (qa/ext), DEEPR flow.** Inputs from Andrey (files/,
  server .tmp/andrey/ext): `hybrid_curated_mrnet_v4_kneemri10_labels.csv` (4,349 competition rows = our train set,
  no gold; 1,250 mrnet-0000..1249 = MRNet train+valid; 745 kneemri-<exam>-<series>) and
  `mrnet_v2_kneemri10_cellw.csv` (external rows only, 0/1; MCL and Baker's 0 for every external study, ACL = the
  datasets' own hard labels). His labels come from a "relatively weak model"; a relabel with his ensemble is coming.
  - *Rule:* competition rows keep v3_pseudo_full exactly (his competition labels Spearman 0.57-0.94 vs ours; using
    them would pull the leg toward A, cf. Opus); external rows = his labels clipped, weight = his cell weight.
  - *Code:* `qa/ext/convert_ext.py` (arrays -> cache v1 entries through preprocess.series_to_volume with an assumed
    160 mm FOV; MRNet sag -> sag_fs, cor -> cor_t1, ax -> ax_fs; KneeMRI -> sag_fs; restricted unpickler),
    `qa/ext/make_inputs_ext.py` (targets v3_pseudo_full_ext, folds_v1_ext.csv, merged cache
    /raid/sroziewski/cache/rsna_kad_ext/v1, OAI recipe), `configs/main_armc_pseudo_full_ext.yaml` (= pseudo_full
    but run_id/targets/folds), `qa/ext/worker.sh`. Tests `qa/ext/test_ext.py` 8/8 for pickle protocols 2/3/4/py2;
    real KneeMRI 745/745 converted (5 are 320x316), loader/targets/folds/index invariants hold. Orientation montage:
    same as competition sag_fs (patella left, femur up); KneeMRI anatomy ~10-20 % smaller in the crop, i.e. true FOV
    probably > 160 mm - FOV to be asked of Andrey. Review: no serious findings; fixed rmtree guard, per-source
    index merge, layout check, duplicate checks, merged index limited to targeted studies (dtype drift), GPU order.
  - *MRNet access (2026-10-16):* Redivis `aimi.mrnet_knee_mri_s:4a2c:v1_0` (Stanford AIMI), user's account
    already had data access; the API record carries no usage text, and the user confirmed the access needed no
    institutional agreement ("no problem with license") - fits the 10-14 ruling, so leg A's MRNet use stands too.
    Downloaded with `qa/ext/mrnet_dl.py` (md5-checked, 3,750 files, 7.2 GiB) to
    /raid/sroziewski/data/rsna_kad/external/mrnet/MRNet-v1.0. External cache: /raid/sroziewski/cache/rsna_kad/ext_v1.
  - *Waiting for:* Andrey's relabelled table.
  - *Judge (fixed now):* 3 seeds 2026/2126/2226; gold-58 is information only (gold under-rates extra-knee models,
    Andrey's MRNet -0.009 gold / +0.003 LB) with a floor: joint with the new C must not lose > 0.004 on gold; then
    one LB probe of the joint; adopt only at LB >= 0.960 (OAI C precedent tied, so a tie rejects).
  - *Training:* KneeMRI FOV switched to 179 mm (Andrey: MRNet 0.62 mm/px ~158 mm, KneeMRI 0.56 mm/px ~179 mm) before
    the final runs; 6,344 train rows, 8 epochs, ~300 img/s, ~2.4 h/seed; done 2026-10-16 18:57 / 18:59 / 21:20 UTC.
    Per-seed gold 0.9185 / 0.9262 / 0.9247 (old C same seeds 0.9216 / 0.9223 / 0.9185).
  - *Gold judge (`qa/ext/gold_joint_ext.py` -> `.txt`, 2026-10-16 21:30 UTC): FLOOR PASSED.* C 3-seed mean 0.9255 vs
    0.9239 old, +0.0016 [-0.0060, +0.0088] P=0.66; JOINT (0.5 v3 + 0.5 Qile) 0.9423 vs 0.9426, -0.0003
    [-0.0030, +0.0020] P(>0)=0.40 (floor -0.004); joint3 form 0.9448 vs 0.9448 (info). Synovitis +0.006, lat men /
    lat OA -0.006. Spearman old ~ ext C 0.977, ext C ~ Qile 0.904 (old 0.900), ~ A 0.900 (old 0.888). Earns one LB
    probe of the joint with ext C (needs a team slot); adopt only at LB >= 0.960.
  - *LB probe (user-requested 2026-10-17):* `qa/ext/export_ext.py` -> export/main_v5_C_ext (the 3 judge seeds at 1/3,
    main_v5_C's wheels/prior) -> private dataset sroziews/rsna-kad-weights-v5c-ext. Kernel sroziews/rsna-kad-joint-extc
    v1 (qa/ext/kernel_joint_extc, build_joint defaults + --partner-fast-glob --require-all-legs, inner meta
    qa/ext/inner_v3ext with only the weights dataset swapped): INNER and PARTNER sources byte-identical to joint v3's
    (qa/qile/kernel_joint_fast). Public run: weights found under rsna-kad-weights-v5c-ext, 3 members, V 452 s / P 79 s,
    0 failures, weights .5/.5. Submitted 2026-10-17 21:33 UTC = **57326594**.
- ***57277456 (joint3 DIAG-ENCODE) scored LB 0.840 = code 3: leg A (hx48) timed out with a budget < 1.5 h*** (seen
  2026-10-17 10:44 UTC; nearest codes 0.878 / 0.797). So V + P used > 7.1 h of the 8.6 h on the hidden test and A
  (~106 min hidden) never had room; A's code and image are fine. joint3 v1's 0.959 was the V/P fallback.
  Team runs the same day: Qile's flat5 v6 (57283695: hx48 .30 + OL-plane-g1-FULL .30 + MT2T-hires-g1-FULL .20 + R7
  .15 + v5-C .05, require-all-legs) **0.959** - all its legs ran, ties joint v2; blend5 v1 (57298855: nartaa S6 .35
  + v5-C .20 + hx48 .20 + OL-plane-FULL .10 + heliosli R6 flip .10, require-all-legs) pending.
- **Public notebooks 2026-10-17 (pulled to server .tmp/pub, read only):** the public top is 0.958, a better version
  of our V-leg base (Raptor 384 0.5 + R7 0.5, LB 0.956): heliosli/this-is-best-public-dataset-new-version-0-9586 and
  goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958 (v25/26 0.958) = nartaa Raptor half (0.65 r384 + 0.35 c224,
  plain + anatomical mirror) + heliosli compact **R6 with flip TTA** (bundle fe055406, dataset
  rostigor/rsna-knee-r6-fe055406), per-finding Raptor share (heliosli: Medial OA .5, Lateral OA .15, PF OA .65,
  Synovitis .85, Baker's .30, else .5; goodpjw: Syn .8, Lat OA .2, PF OA .6, Baker's .35) + goodpjw's own 2.5D
  ConvNeXt reader at 1/10. goodpjw v4: adding S6 next to R6 cost 0.001. R7 at 0.4 of the compact slot scored the same
  0.958. lavinwins "gold medal" = the older 0.943 DINOv2/A5/RadImageNet/Raptor stack (v6 reweighting, no new models).
  leoprovorov's write-up: LB sd ~0.003 on ~640 test studies (> the spread of the top 400 teams); ceiling ~0.96 at
  error correlation 0.86-0.89. Kaggle discussions are not reachable from the CLI and were not read.
  Trap: heliosli/rsna-knee-r6-frozen-fe055406 (latest version) holds the R7 bundle byte for byte (manifest bundle_id
  8c7b666e), so heliosli's notebook would fail its own `bundle_id == fe055406` assert; the real R6 is
  rostigor/rsna-knee-r6-fe055406 (different fold weights, bundle fe055406).
- **r6base: 0.958 recipe as our V-leg base, user-requested 2026-10-17 (qa/r6base). REJECTED, nothing submitted.**
  qa/v2stack/build_kernel.py gained `--generic-nb` (any notebook layout: every %%writefile cell becomes a file write,
  other code cells verbatim but NB_T0; default, v3stack and --fake v3stack builds byte-identical to before, v3stack ==
  the submitted kernel md5 36e1cd4b), qa/v2stack/e2e.sh `EXTRA_DS`. Base = goodpjw2008's notebook v27 (sha256
  a906df93..., qa/r6base/src; image 37c64f7d, versioned R6), main_v5_C 0.3; kernel qa/r6base/kernel_blend
  (sroziews/rsna-kad-v5c-r6stack, not pushed). e2e qa/r6base/e2e_r6.sh: 58/58, 0 failures, 0 fallbacks, Part B ran;
  kernel == 0.7 rank(base) + 0.3 rank(C) exactly. Verdict qa/r6base/gold_r6.py -> gold_r6.txt (information only):
  base 0.9384 vs R7 base 0.9356 (+0.0028, P=0.82); V leg 0.9398 vs 0.9403 (-0.0005); **joint v2 form 0.9424 vs
  0.9441, -0.0017 [-0.0062, +0.0023] P(>0)=0.21** (Lat Men -0.012, Synovitis -0.008; MCL +0.006). Spearman R7 ~ new
  base 0.986: the new base moves toward Qile (0.932 vs 0.924), so its gain vanishes in the joint. Runtime: V leg
  650 s vs 532 s on the A6000 (x1.22; R6 GPU 5.3 vs 2.8 s/study with flip TTA, + 224 stage); with V + P >= 7.0 h
  hidden (57277456), x1.22 on the V share leaves no margin in 8.6 h. Both point the same way: keep joint v2.
- **P+A: Qile .6 + Andrey hx48 .4, no V leg (qa/pa), user-approved 2026-10-17. Submitted 57325519.**
  Gold-58 combos (qa/r6base/gold_combos.py -> gold_combos.txt, vs joint v2 0.9441): P+A .6/.4 0.9477 (+0.0035,
  P=0.87), .5/.5 0.9466, joint3 0.9462. joint_template.py drops weight-0 legs from the run order (`--w-partner 1.0`
  = V not run); qa/joint/test_require_all.py case pa_on, 7/7 + test_diag_encode 10/10 PASS. Kernel
  sroziews/rsna-kad-pa v1 (joint md5 01395b9d, P and A sources byte-identical to joint3's, image 37c64f7d,
  require-all-legs). Public run: P rc 0 112 s, A rc 0 68 s, weights_used {P .6, A .4}, total 180 s. Hidden runtime
  ~ P + ~106 min A, well under 8.6 h. Final pick only at LB >= 0.959.
- **Scan 2026-10-18 (qa/discussion_2026-10-18: run.sh, crawl8.py -> new_since_1015.txt, kernels_*.txt; kernels
  pulled to server .tmp/pub2).** Nothing above 0.958 public; nothing worth a slot.
  - *Code:* every top notebook is the heliosli/goodpjw 0.958 recipe (Raptor 384 + 224 at 0.35, R6 flip, rank 50/50
    with hand weights Raptor share Medial OA .5 / Lat OA .2 / PF OA .6 / Synovitis .8 / Baker's .35; goodpjw reader
    per finding 0.003-0.035, not a flat 1/10). goodpjw v8 measured: S6 next to R6 0.957, gold-58-fitted per-finding
    weights 0.957 / 0.955 ("do not tune per-finding weights on the 58 gold studies"). laimerbotrolenk V33: 224 rank
    half plain / half mirror (unscored). New datasets only nartaa/rsna-knee-s6fall-weights and the goodpjw reader,
    both neutral or negative next to R6. notfu4y "deterministic report" = regex labels + DINOv2-S, no score.
  - *Forum:* hengck23 (10-15, last post before the final-week silence): rajpurkarlab/3DReasonKnee on Hugging Face =
    OAI knees with boxes/segmentations for most of the 12 findings (licence unchecked; too late to train on).
    Danial Zakaria released the S6/CoAtNet training code (Apache-2.0). OAI fairness thread (743416): geo-blocked
    OAI access raised again, host will not rule per dataset; teams post lists of external weights and their source
    data (nishchal jain 10-18) - our picks contain Raptor (OAI), R7 (fastMRI aux) and, via joint3, Andrey's
    MRNet/KneeMRI models, to be disclosed in any write-up.
- **GLM-5.3-Flash vision as labeller for the external knees, user-requested 2026-10-18: REJECTED.** MRNet/KneeMRI have
  no reports, so an LLM could only label them from images. Endpoint <llm-host>:8000 (`opi-glm-4.6`, root
  zai-org/GLM-5.3-Flash) accepts images. A 2026-10-04 pilot already existed (qa/glm_vision/report.md, run.py: 8 fat-sat
  slices; effusion 0.806, lat men 0.553, lat OA 0.399 vs ours 0.83-0.99; not a usable teacher), undocumented here
  until now. New pilot `qa/glm_vision/pilot.py` (3 plane montages of 12 slices, competition definitions, guided JSON,
  gold used only for scoring) on gold-58 vs main_v5_C on the same studies: thinking off 53/58 answered, macro AUC
  **0.659 vs 0.919** (ACL 0.594, menisci 0.54/0.51); thinking on (pilot_think.py, ~24 s/study) 47/58, **0.729 vs
  0.925**; 0.9 C + 0.1 GLM = 0.917 / 0.925, no gain (pilot_nothink.txt, pilot_think.txt). Far below Andrey's model
  labels (~0.937 gold); external-knee relabelling with a VLM closed.
- **joint3 q3: joint3 with Qile's faster notebook, user-approved 2026-10-17. Submitted 57327062.** Qile (17 Oct):
  decode overlapped with GPU, ~1.9x faster, same predictions, his part <= 2.5 h on the hidden test; that should free
  ~2 h for A, which timed out in 57277456. His new version (qilejiang76/rsna-knee-ours-mtr1-mth4-olp5, sha256
  9657e14c..., server .tmp/qile/v3nb; same datasets/image) has its own os.walk input finder, so no
  --partner-fast-glob. Parity `qa/qile/e2e_qile_v3.sh` (leg P alone via --w-partner 1.0; his datasets re-downloaded
  to /raid/sroziewski/data/rsna_kad_ext/qilejiang76 and mounted as real dirs, his walk does not follow symlinks):
  gold-58 P == old P exactly (max abs diff 0), 176 s, peak RSS 5.9 GB. Kernel sroziews/rsna-kad-joint3q3 v1
  (qa/andrey/kernel_joint3q3): V, A, weights, datasets, image identical to kernel_joint3_diag, only P changed,
  require-all-legs, no diag. Public run: V 467 s, P 55 s (old 85 s), A 63 s, weights .4/.4/.2. A score proves all
  three legs ran on the hidden test (gold 0.9462 vs joint v2 0.9441); an error means a leg still drops.
  - ***LB 2026-10-18: 57327062 = 0.959 (all three legs ran), 57326594 ext C = 0.959 (tie, not adopted), 57325519
    P+A = 0.958 (not a candidate).*** Team runs: flat7 57332429 0.959, Andrey's ConvNeap 57361733 0.943.
  - *Equal thirds (user-approved 2026-10-18):* kernel sroziews/rsna-kad-joint3q3eq v1 (qa/andrey/kernel_joint3q3eq)
    = joint3q3 with A at 1/3 (joint.py diff: only that weight; metadata only the id). Public run V 505 s / P 59 s /
    A 65 s, weights 1/3 each. Submitted 2026-10-18 21:02 UTC = **57368898**; adopt only at >= 0.960 (gold 0.9456 vs
    0.9461). ***LB 0.958: not adopted.*** Team: Qile flat6 57370104 0.959, peach nested958 57372562 0.958.
- **S6 retrain on our labels (s6r, user-requested 2026-10-18; REJECTED 2026-10-19).** nartaa's released S6 all-data
  trainer (nartaa/rsna-knee-s6fall-weights training_code/s6, Apache-2.0) run unchanged except the targets: our
  v3_pseudo_full target + cell weight as arm OURS=bce:prior:w (qa/s6r: setup.sh, make_targets_s6r.py, train.sh,
  chain.sh). 4,349 studies, 16 epochs, convnext_small, 8.0 h on one GPU; loss 0.489 -> 0.403; model
  .tmp/s6r/runs/full/models/OURS_s42_fall.pt (sha256 8512fadb...). Inference: qa/s6r/leg_s6r.py (= leg_s6.py with our
  constants; run_config "arms" compared on the name before "="), e2e qa/s6r/run_gold_r.sh, 141 s for gold-58.
  Gate qa/s6r/gold_blend.py (joint3 .4/.4/.2 + S6 at a fixed 0.15): ours alone 0.9283, blend 0.9454 vs joint3 0.9462
  (-0.0008, CI [-0.0046, +0.0027]), spearman to joint3 0.948; nartaa's checkpoint alone 0.9314, blend 0.9456 (-0.0006),
  spearman 0.923. Pre-set rule was "only if it beats nartaa's checkpoint": it does not (our labels make it more
  correlated with joint3, so it adds less). No dataset, kernel or LB probe.
  - *User override 2026-10-19 ("gold is not necessarily a good predictor"): LB probe anyway.* Dataset
    sroziews/rsna-kad-s6r (qa/s6r/make_ds.sh: model + run_config + DONE + nartaa's code_manifest.json/s6_extras.bin,
    our checkpoint-manifest.json). Kernel sroziews/rsna-kad-joint3s6r v1 (qa/s6r/kernel_joint3s6r): joint3q3 + extra
    leg S6 = leg_s6r.py, --w-partner 0.5, A 0.17, S6 0.15 -> V .34 / P .34 / A .17 / S6 .15, require-all-legs; P and A
    sources byte-identical to joint3q3's. v1 public run all legs ok (V 473 / P 60 / A 68 / S6 101 s). v2 = A swapped
    to hx48 v6 (cuts hidden runtime for the S6 leg): public run V 478 / P 66 / A 46 / S6 96 s, weights as planned.
    **Submitted v2 2026-10-19 17:05 UTC = 57403485**; adopt only at >= 0.960.
  - joint3v6 public run: V 552 / P 78 / A 54 s, .4/.4/.2, all legs ok: the runtime-safe stand-in for the joint3
    final pick (not probed: same predictions).
- **hx48 v6 in joint3 (user-requested 2026-10-19).** Andrey: v6 = same checkpoints/preprocessing, faster (~50 min
  hidden). Pulled andreytikhomirov/rsna-knee-native-hx48 (latest, .tmp/andrey/v6pull, notebook sha256 3455278...;
  image 2757e0c7, ours stays 37c64f7d); its cell 0 writes hx48_ensemble.py, extracted with ast to
  qa/andrey/hx48_ensemble_v6.py (sha256 7ace8477...): the 7 old embedded sources byte-identical + infer_fast.py (one
  pass), same HX48_WORK interface, no network code. His gold58 v6 output vs our joint3 A leg: max diff one rank step
  (1/58), min per-finding spearman 0.9993. Kernel sroziews/rsna-kad-joint3v6 v1 (qa/andrey/kernel_joint3v6) = joint3q3
  with only the A source swapped (.4/.4/.2, require-all-legs). Pushed, queued.
- **Inference blur on main_v5_C, gold-58 (user-requested 2026-10-19 after Andrey's report: Gaussian sigma 0.5-0.8 helps
  hx48 +0.0018, MT-hires +0.0007, S6 / nartaa CoAtNets / Qile's blend; Synovitis/Contusion/Baker's up, Fracture down).**
  `qa/blur/blur_gold.py` (read_triplets patched: blur each cached uint8 slice before warp and before the model's
  normalisation; 6 seeds, probability mean; 'none' == the runs' own gold preds, max |d| 1.2e-7), `qa/blur/report.py`
  -> qa/blur/report.txt. Variants fixed before scoring. none 0.9255; Gaussian 0.5 -0.0007, 0.65 +0.0004, 0.8 +0.0001;
  Student-t nu 3 @0.65 -0.0004, nu 5 @0.65 +0.0004; every CI ~ +/-0.004. Same per-finding pattern as Andrey
  (contusion +0.004..+0.012, pf_oa +0.005, synovitis up; lat_oa -0.008..-0.015, med_men -0.005). **No effect on C**;
  Student-t no better than Gaussian. Our cache is already percentile-windowed and area-resized to 384, which may
  already smooth what blur removes.
- **joint3 + blur in legs P and A (DEEPR, user-requested 2026-10-19).** `qa/blur/patch_legs.py` writes patched copies
  (originals untouched): the embedded source literal (Qile: the ours_infer.py text of cell 0; hx48 v6:
  FILES["infer_fast.py"]) is found with ast and replaced in place by repr() of the patched text. Blur on the model-input
  grid, after the resize and before normalisation, every finding, sigma 0.65 (Andrey's 0.5-0.8 midpoint; one constant
  for his sweep). P: cv2.GaussianBlur per uint8 crop slice after decode()'s warp (the one crop builder), rounded to
  uint8. A: separable torch Gaussian over the uint8 window batch before gpu_intensity (the one call; outside autocast;
  gpu_intensity takes float), kernel size = cv2's float rule. Rejected: Fracture left unblurred (a second pass, 2x the
  runtime); run-time monkeypatching (the legs run as subprocesses). Tests `qa/blur/test_patch_legs.py` 43/43 (diff =
  inserts only; sigma 0 identity; blur == cv2 within 1e-4; non-ASCII / CRLF / ! % lines; anchors once). Reviewer: no
  blocking bug; fixed: value read from the unmasked bytes, kernel-size formula, single call sites asserted, test-only
  GPU-map rewrite enforced and kept out of kernel sources, build errors fatal, noise-floor case orig2. Gold e2e
  `qa/blur/e2e_blur.sh` + `score_legs.py` (fake tree, P + A, cases orig / orig2 / s0 / s0.65, GPU 1):
  parity P s0 == orig exactly; A s0 vs orig 3.45e-2 = orig vs orig2 (hx48 is not bit-deterministic on GPU, spearman
  0.9998). Gold-58 (information only): **leg P 0.9399 -> 0.9430 (+0.0032), leg A 0.9367 -> 0.9389 (+0.0022; Andrey
  +0.0018), joint3 0.9462 -> 0.9459 (-0.0003, CI +/-0.003)** - the leg gains do not survive the blend. Runtime on gold:
  P 222 s unchanged (decode overlaps the GPU), A 98 -> 124 s (+26 %; criterion < 10 % not met for A alone; hidden
  ~ +13 min on ~50 min). Kernel sroziews/rsna-kad-joint3blur v1 (qa/blur/kernel_joint3blur, sources qa/blur/src_s0.65)
  = joint3v6 except the P and A sources (P differs in the literal line only, A in the FILES line only, joint.py in the
  two blobs + docstring). Public run V 830 / P 117 / A 110 s, all legs ok. **Submitted (user-requested, before
  Andrey's sweep) 2026-10-19 18:18 UTC = 57405419**; adopt only at >= 0.960.
- **Andrey's blur sweep (2026-10-19, his artifact "Knee MRI blur sweep", screenshots read):** 8 levels 0-0.504 mm,
  gold-58, 90 % paired bootstrap. Clear of zero at 0.288 mm: **R7 +0.0101 (P 99.8 %)**, MT-hires +0.0067, S6 +0.0058;
  within noise: hx +0.0034, Qile blend +0.0029, hx48 ensemble +0.0009, nartaa-1 +0.0009, OL-plane, 48-pos; v5-C
  -0.0005, MT-r50 -0.0038. Synovitis rises almost everywhere (R7 +0.058); Lateral OA / Fracture can lose. Caveat:
  best of 8 levels x 11 models, so peaks are optimistic; R7's curve is smooth over neighbouring levels.
- **R7 blur inside V (DEEPR, 2026-10-19).** `qa/blur/patch_r7.py`: in the v3stack base notebook's cell 2,
  SOURCES["probe_infer.py"] gains `_k1_blur_install(native)` in install_native, wrapping
  training.dicom_geometry.pixels_from_dataset (after modality LUT / MONOCHROME1, before canonicalize, quantisation,
  window): cv2.GaussianBlur sigma 0.288 mm / PixelSpacing per axis (row -> y, col -> x), frame-wise; SOURCE_SHA
  updated. Only R7 decodes through it (Raptor and C have their own readers in their own processes; R7's side
  classifier and anchors read the blurred series, as in Andrey's setup). Tests `qa/blur/test_patch_r7.py` 29/29.
  Reviewer: blur correct on every R7 decode path; fixed in the harness: fallback/scored check (a blur crash would
  read as "blur hurts"), Raptor/C-unchanged check, kernel_test save/restore with trap + md5, stale-run removal,
  negative sigma rejected. Gold e2e `qa/blur/run_r7.sh` + `score_r7.py` -> .tmp/blur/r7/score_s0.288.txt: R7 58/58
  scored, 0 fallbacks both runs; sigma 0 == unpatched for R7 / base / V exactly; C unchanged, Raptor 5.6e-5 (GPU).
  **R7 0.9312 -> 0.9411 (+0.0099, 90 % [+0.004, +0.017]; Andrey +0.0101); base +0.0038; V 0.9403 -> 0.9416
  (+0.0014); joint3 + R7 blur +0.0005; joint3 + R7 + P/A blur 0.9462 -> 0.9472 (+0.0010, P 0.71); vs the pending
  P/A probe +0.0013 (P 0.87).** V inner qa/blur/v_inner_s0.288 (build_kernel, no --fake; vs the shipped v3stack inner
  only the generated header, SOURCE_SHA and SOURCES lines of the base differ). Kernel sroziews/rsna-kad-joint3blurall
  v1 (qa/blur/kernel_joint3blurall) = joint3blur with only the V inner swapped. Public run V 784 / P 111 / A 93 s,
  all legs ok. **Submitted 2026-10-19 21:07 UTC = 57409437** (user-approved). Adopt only at >= 0.960.
  Andrey (same evening): flat7 v5 + 0.288 mm blur on R7 and Qile's two high-res models, gold 0.9485 -> 0.9503; one of
  the remaining slots offered to him.
- **LB results 2026-10-20** (Oct 19 probes):
  - 57405419 joint3v6 + P/A blur: 0.960.
  - 57403485 joint3 + S6r: ERROR (unhandled error on the hidden test).
  - 57409437 joint3blurall: ERROR. Same P/A blur as 57405419, so our R7 patch (`patch_r7.py`, `pixels_from_dataset`
    wrap) is the suspect on hidden DICOMs.
  - Andrey A 57411614 flat7-blur: 0.960.
  - Andrey B 57411322 j3-stack-blur: **0.961**, the first one above 0.959. His R7 blur (DICOM-level hook) survived the
    hidden test.
  - The two ERRORs are most likely leg timeouts, not crashes:
    - require-all-legs turns a leg timeout or skip into the same SystemExit as a crash.
    - joint3 already uses > 7.1 h of 8.6 h on V + P (57277456).
    - S6 runs last.
    - The R7 blur adds CPU time inside V.
    - R7 decodes each slice in try/except, so a blur exception would not crash the run.
    - Andrey's identical 0.288 mm R7 blur scored in flat7-blur and B.

    Not proven: the hidden log is not visible.
  - **Rule from 2026-10-20 (user): every LB probe built by build_joint uses `--diag-encode` and drops
    `--require-all-legs`.** The REQUIRE_ALL exit runs before DIAG_ENCODE, so with both flags set a failure is still a
    bare error. Final picks are never diag-encoded.
- **joint3blurall2 (DEEPR 2026-10-20), the fix for 57409437.**
  - *Research:*
    - The R7 blur cannot fail V: any R7/base failure leaves V = C only.
    - Blur cost is 0.5-1 ms per slice, about 0.1 s per study.
    - Gold R7 ran 335 vs 332 s with the same side flips (1) and 58/58 scored.
    - V and P are byte-identical to 57405419 (0.960).
    - So the dropped leg was almost surely the last one (A: timeout or skip): the run sits at the 8.6 h edge.
    - Andrey's kernels have no Raptor, no per-leg timeouts and a sitecustomize pydicom hook.
  - *Fix:*
    - A unblurred: hx48 v6 as in joint3v6, gold -0.0006 [-0.0022, +0.0010], about -13 min hidden.
    - `build_joint.py --total-h` (default 8.6, max 8.85), set to 8.8 h: +12 min for P and A. The inner keeps its own
      8.6 h clocks.
    - `--diag-encode` without require-all.
  - Gold with every blur combination: all within 0.9459-0.9472.
  - Build script `qa/blur/build_blurall2.sh`; kernel sroziews/rsna-kad-joint3blurall2, joint.py md5 08fd5354.
  - Tests:
    - `qa/joint/test_total_h.py` 14/14: stub legs; the same slow A times out at 8.6 h and runs at 8.8 h.
    - `qa/blur/test_blurall2.py` 26/26: byte-identical rebuilds, legs equal their sources.
    - test_require_all 7/7, test_diag_encode 10/10.
  - Review fixes: `ap.error` instead of assert, clearer help text.
  - Public run v1: diag code 0, all legs (V 1235 / P 182 / A 159 s; V varies 552-1235 s across identical-V runs, i.e.
    machine noise). **Submitted 2026-10-20 09:43 UTC = 57432197**; adopt only at >= 0.960.
- **Andrey's team-submissions page (2026-10-20), DEEPR "act on it".**
  - *Leaderboard 10:01 UTC:*
    - We are rank 82/6,264 at 0.961.
    - Medal cuts: gold rank 22 = 0.964, silver rank 312 = 0.958.
    - Teams per score: 0.961 = 25, 0.962 = 16, 0.963 = 19, 0.964 = 12.
    - Deadline 2026-10-22 23:59 UTC; 14 slots left from 10-20.
  - *LB vs gold-58 (54 subs):*
    - Above gold 0.94 the slope is 0.24 (residual SD 0.0008).
    - So a +0.001 gold gain is about +0.0003 LB, under the LB noise of about 0.003.
    - Gold can only veto; it cannot find a +0.003.
  - *Hidden time (queue included):* flat7-blur 6.3-6.5 h, B 7.3-7.4 h, joint3blurall 8.4-8.6 h (no score; consistent with
    the 8.6 h joint budget), joint3v6+blur <= 11 h.
  - *Gold-58 rebuild of B* (`.tmp/research_gold/repro_b.py`, members r7 blur, MTR/MTH/OLP, v5c, hx48 v6 + Andrey's
    embedded stacker): 0.9504 vs his 0.9509. The missing piece is MTH with his 0.8 px volume blur.
  - *Variants screened* (`variants.py`, delta vs B, 90 % interval):
    - Qile part = our P blur 0.65 px on all views: +0.0012 [-0.0010, +0.0036], P 0.81.
    - hx48 blur: -0.0006.
    - Raptor 0.05/0.10/0.20: -0.0001/-0.0005/-0.0020.
    - Our S6: <= -0.0001.
    - V: <= 0.
    - C 0.05: +0.0001, larger weights negative.
    - Blur/no-blur rank averaging: r7 -0.0004, P +0.0007, P+A +0.0003 (all below full blur).
    - Our members add nothing to B.
  - *Built B-Pblur* (`qa/bblur/build_bpblur.py`):
    - Andrey's B kernel (pulled, md5 62df89a0) with SRC['ours'] = patch_legs.patch_qile_src(run_ours, 0.65)
      and HIRES_BLUR_PX 0.8 -> 0.0 (no double blur on MTH).
    - Only those two lines change (asserted in the build).
    - Kernel sroziews/rsna-kad-bpblur v1 (bpblur.py md5 32cf28a9), pushed.
    - Tests `qa/bblur/test_build_bpblur.py` 27/27, blur suites 72/72.
    - Review: no blockers; added in-build checks, LF writes and a Python >= 3.8 guard.
  - *Open:*
    - Confirm with Andrey that the pulled B is the version that scored 0.961 (he is refitting stackers today).
    - Submit only after the public run passes and the user approves.
  - Open, not fixed:
    - A timed-out leg's grandchildren survive (`subprocess.run` kills only the child).
    - An R7 failure inside V is invisible to diag codes.
  - *bpblur public run (3 studies):* COMPLETE, 2,398 s, nothing dropped (v5c 1,333 / r7 499 / ours 262 / hx48 302 s).
    All four Qile members differ from B's public output (mean |d| MTH 0.016, MTR 0.034, OLPC 0.017, OLPS 0.022), so the
    patch is live. Submitted 2026-10-20 13:32 UTC as 57439962 after Andrey's OK (kernel v1); adopt only at >= 0.962
    (beats B).
  - *Andrey's mth-blur-oof run on our server* (`qa/andrey/run_mthoof_local.sh`, screen mthoof, GPUs 1,3, about 27 min):
    - `.tmp/mthoof/fake/kaggle/working/mth_oof_blur_all.csv`: 4,349 rows (all non-gold, unique IDs), 5 folds
      (about 300 s each), no NaN.
    - Parity against Qile's `member_mt_hires.csv` on 40 fold-0 studies (sharp): Spearman 0.865-0.973 per finding,
      mean |d| 0.034, max 0.48. The same code matches Andrey's Kaggle B output to 0.0012, so Qile's OOF file comes from
      another evaluation path (asked Qile).
  - *Andrey's r6-r7-blur-oof (asked 15:31) on our server:* pulled to `.tmp/andrey/r67oof` (read: the wrapper and the
    two embedded legs, run_r7 = the B leg, run_r6 = the flat4+R6 leg with flip TTA off; blur 0.288 mm hook; hash folds
    sha1(uid) % 4 = heliosli's compact_train folds, not folds_v1; fold model k only, on fold k + 12 in-fold checks).
    `qa/andrey/prep_r67oof.py` remaps /kaggle/ and '/tmp (wrapper + both legs re-packed) and sets MODELS to one model;
    `qa/andrey/run_r67oof_local.sh` runs it in the throwaway `.tmp/v2stack` venv (the legs pip-install pinned wheels;
    the wrapper + nvidia-smi shim keep the controller on one GPU). Screens r7oof (GPU 3) and r6oof (GPU 1), started
    15:39 server time; outputs under `.tmp/r67oof/<m>/fake/kaggle/working/`.
    - Done: R6 8,250 s, R7 8,609 s (1.8-2.1 s/study wall per fold); each `<m>_oof_blur_all.csv` 4,349 rows, 0 unscored,
      folds 0-3, no fallbacks. Handoff copy (OOF csvs, per-fold csvs, run logs, leg logs) in
      `.tmp/for_andrey/r67_oof_blur/` (local).
- **LB 10-20 afternoon:** Andrey submitted 57438367 "flat4+ R6" (gold 0.9525, about 5.6 h, require-all-legs); pending.
  57432197 joint3blurall2 = **0.960** (seen 17:28 UTC): diag code 0, all three legs ran inside the 8.8 h budget, so
  the 57409437 error was the budget. Ties A and 57405419, below B 0.961: not adopted, finals stay B + A.
- **LB by 10-21 02:45 UTC:** 57439962 bpblur = 0.961 and 57438367 flat4+R6 = 0.961, both tie B (not adopted; a tie
  is within the about 0.003 LB noise). Andrey also used the last two 10-20 slots: 57452940 "D flat5" and 57455318 "C3S"
  (C3 + Synovitis stacker), both pending. 10-20: 5/5 used; 10-21: 5 left.
- **DEEPR 2026-10-21 "anything left to push higher?" Verdict: nothing new to build; spend slots on Andrey's stacker
  refit (blurred OOF) and pick finals for diversity.**
  - *LB 02:58 UTC (6,302 teams):* we are rank 82 at 0.961. Gold cut rank 22 = 0.965 (12 teams at 0.964, 15 at
    0.963, 18 at 0.962, 26 at 0.961). Silver cut rank 315 = 0.958, where 1,454 teams sit (the public notebook).
    The API gives 3 decimals only, so ties can't be ordered.
  - *Forum (research_disc):* public LB = ~30 % of 1,322 hidden studies, SD of one submission ~0.003. Staff: public
    blends tend to drop on private. Submission queue is 4-6 h since 10-17.
  - *Code (research_pub):* nothing above 0.9586. All 0.958+ notebooks use Raptor 384/224 + R6 flip + per-finding
    hand weights + goodpjw reader. All of these were already screened here: reader LB tie (0.948); 0.958 base in V
    -0.0017 gold; Raptor on B -0.0001..-0.0020; S6 negative. New datasets (neronrx 4-fold ConvNeXt, tensorhydra OAI
    Raptor, V-JEPA, Engine11) have no public score and no time to validate.
  - *flat4-syn-r6 = 57438367 (C3):* R6 no-flip blur, MTH blur 0.8, OLP, hx48 at .25 each + Synovitis stacker 25 %;
    no v5c/MTR; ~5.2 h. Leg sources other than r7/R6 byte-identical to B.
  - *Picks proposal:* B 57411322 + C3 57438367 (both 0.961, most different members: R7 + v5c + MTR + 12-finding
    stacker vs R6 flat + Synovitis stacker) instead of B + A (0.960).
- **bpsyn = bpblur + flat4-syn-r6's Synovitis stacker (DEEPR 2026-10-21, user request).**
  - `qa/bblur/build_bpsyn.py`: inserts flat4's stacker block (JSON literal byte-identical, math as flat4 L243-249)
    before bpblur's `say('stacker5 blend ...')`; Synovitis = .75 rank(bpblur) + .25 rank(stacker on MTH/OLP/hx48
    ranks); the other 11 columns unchanged. bpblur md5 32cf28a9 -> bpsyn.py d045e249; kernel sroziews/rsna-kad-bpsyn v1
    pushed (public run started).
  - Tests `qa/bblur/test_build_bpsyn.py` 20/20 + bpblur 27/27 (tail executed on synthetic members; mutation check).
  - Gold-58 (`qa/bblur/bpsyn_gold.py`, info only, stacker fed unblurred per-member MTH/OLP): bpblur 0.9515, bpsyn
    0.9514, d -0.0001 [-0.0012, +0.0009] P 0.42; Synovitis AUC 0.8769 -> 0.8757 (stacker alone 0.8501).
  - Review: no code bugs. Note: stacker fitted on unblurred OOF; in bpsyn both MTH and OLP are blurred 0.65 px
    (flat4 blurs MTH 0.8 only), so it is not flat4's stacker on flat4's inputs. Fixed: gold script reads share and
    finding from the JSON.
  - Expected LB effect <= 0.001 (one finding of 12). User (10-21, going to sleep): "make submission after all" =
    submit once the public run passes. Public run: all 6 members, dropped none, 1,648 s; submitted 2026-10-21
    03:33 UTC as **57463786**; adopt only at >= 0.962.
- **LB 10-21 07:34 UTC:** 57452940 D flat5 = 0.961, 57455318 C3S = 0.961 (both tie B, not adopted); 57463786 bpsyn
  still pending. Live top: 0.969. 10-21: 1/5 used.
- **LB 10-21 14:43 UTC:** 57463786 bpsyn = **0.961** (ties B, not adopted). Andrey submitted 57485680 "U2" 14:19 UTC
  (D flat5 + hx48 crop warp anti-aliased; gold58 0.9526; finals only at >= 0.962); pending. 10-21: 2/5 used.
- **Selected on Kaggle (seen 10-21):** B 57411322 j3-stack-blur (0.961) + A 57411614 flat7-blur (0.960). Proposal
  (not agreed): swap A for flat4-syn-r6 57438367 (0.961, R6-based, more diverse from B than R7-based A).
- **Stacker refit on blurred OOF for bpblur's config (research 10-21): BLOCKED.**
  - Needs 0.65 px OOF for MTR/MTH/OLPS/OLPC: feasible (Andrey's mth-blur-oof driver + run_ours_patched.py,
    KNEE_MODELS=MTR,MTH,OLPS,OLPC, KNEE_HIRES_BLUR_PX=0, OLP merged as bpblur.py:185-189; est. 25-50 min).
  - But: no fit code (STK JSON has lr/ranklr per finding, no intercept/C; "ranklr" undocumented), hx16 cv5 OOF not
    on the server; labels = `.tmp/andrey/ext/hybrid_curated_mrnet_v4_kneemri10_labels.csv` (> 0.5).
  - **Fold-split warning:** Qile's g1 models are probably NOT on our folds_v1. Sharp parity: only 11/40 folds_v1
    fold-0 studies match Qile's member_mt_hires.csv within 0.003, the other 29 differ 0.03-0.48, matches and misses
    share site_keys (folds_v1 is GroupKFold by site); ~1/5 match = an unrelated split. Then the folds_v1-driven
    "OOF" (mth_oof_blur_all.csv sent to Andrey) is ~4/5 in-sample and must not be used to fit a stacker. Only Qile
    can confirm his split; his run_ours comment says "folds_v1.csv" (his own file may differ from ours).
    R6/R7 OOF unaffected (heliosli's hash folds, as trained).
  - **Resolved 10-21 (Qile):** g1 models trained on folds_v1, gold excluded; his member_mt_hires.csv comes from his
    training-script eval. His guess (Kaggle code averages all 5 folds) does not apply to our run: every log says
    "folds [k]" / "1 folds/seeds averaged [k]" (parity and f0-f4). So mth_oof_blur_all.csv IS out-of-fold; the
    in-sample warning is withdrawn. The 29/40 parity misses are an eval-path difference, Qile is looking into it.
- **bprefit result (10-21 22:26 CEST): REJECT, nothing submitted.** Unblocked by Andrey's `fit_stacker5.py` + hx16 cv5
  OOF (`.tmp/andrey/stacker5/`). `qa/bprefit/run_qoof.sh` reran MTR/MTH/OLPS/OLPC at 0.65 px after the warp, folds_v1
  out-of-fold (4,349) + gold all-fold, rc 0; `qa/bprefit/run_fit.sh` (log `logs/bpfit_20261021-222439.log`): control
  refit on Qile's original members reproduces B's STK (max |coef d| 1.5e-08) and screens d +0.0000; refit on the
  blurred members: bpblur gold 0.9518 -> bprefit 0.9522, d +0.0004 [-0.0013, +0.0022] P>0 0.66 (Lat OA +0.006,
  Contusion +0.003, PF OA -0.003). Below the +0.001 bar set before the run; a slot would buy a 0.961 tie. Closed.
- **FINALS ticked (seen 10-21 evening, user): B 57411322 + C3S 57455318** (both 0.961; C3S per Andrey's pre-registered
  rule "replaces C3 unless below 0.961"). Only change left: U2/U5 >= 0.962 replaces B. DEEPR 10-21 evening: nothing
  buildable by the 10-22 23:59 UTC deadline clears the slot bar (gold >= +0.001 P >= 0.8, <= 8 h hidden), no slots of ours.
- **LB 10-21 23:35 UTC:** U2 57485680, U1b 57487018, U5 57487526 = 0.961 (all tie; none >= 0.962, B stays);
  C3SE 57491297 pending. Finals unchanged: B + C3S.
- **LB 10-22 07:32 UTC:** C3SE 57491297 = 0.961 (tie, not adopted). All pending scored; finals final: B + C3S.
- **FINALS ticked 10-22 ~20:20 UTC (user, per Andrey): U8 57519548 + C3S 57455318** (both 0.961). U8 = C3S + S48
  OOF stacker on 6 findings at share .25; Andrey: +10 LB positions over our other 0.961s (full-precision rank).
  C3SE dropped (below C3S in the UI order, its pre-registered rule). U4b 57518756 / U7 57521316 COMPLETE with no
  public score. Open: U9 57533566 pending, replaces U8 only if it ranks higher before 23:59 UTC.
- **LB 10-22 23:34 UTC (last check before the deadline):** U9 57533566 = 0.961 ("matching your best", team rank
  unchanged), not swapped in. Team row pasted by the user: rank 87, silver medal zone, 0.961, 135 submissions.
  **Finals at the deadline: U8 57519548 + C3S 57455318.** Write-up: `summary/`.
- **Final standings (user, 10-23): place 85 of 6,425 teams, public LB 0.961.**

## 5. Budgets and risks

| Resource | Item | Estimate |
|---|---|---|
| GPU-h (A6000) | LLM labelling 2 variants + reruns 4 h · MVP 1 h · arm A 15 h · arms B+C 35 h · efficiency 3 h · stage 9 30 h · OOF/TTA passes 5 h | **~95 GPU-h** on 2 GPUs over 4 weeks (fits; peak 2 concurrent jobs) |
| GPU memory | vLLM Qwen3-32B-AWQ ~20 GB + KV (GPU 0, `gpu_memory_utilization=0.6`); training bs 4 @384 ~30 GB (fits GPU 1's 38 GB; fall back to bs 3) | |
| CPU | cache build 24 procs ~15 min; dataloaders 12 workers per training job (2 jobs = 24 cores) | 48 cores shared |
| Disk /raid | cache v1 ~108 GB (≤ 144 GB worst case) · LLM weights 20 GB in HF_HOME · timm weights 1 GB · checkpoints (0.2-0.4 GB × 5 folds × ~6 arms/variants) ~10 GB · label/target/OOF tables < 1 GB | **~140 GB** vs 150 GB budget; v2 cache replaces v1 |
| Kaggle | main: ~2.9 h + 25 % → 3.6 h of 9 h; efficiency ~10-12 min; dataset ~1.5 GB (3 arms × 5 folds + wheels) | |
| Calendar | 34 days: stages 0-4 = 7 d, 5 parallel, 6-9 = 16 d, 10 = 2 d, slack 2-3 d | |

Top risks and mitigations:

1. **Label noise / extractor below target** - three voters with disagreement → uncertainty, severity-aware
   targets, fold-local calibration, gold-58 gate at stage 5 (≥ 0.85) before any 384-px training; if the LLM misses
   the gate, ship the MVP on rules and iterate prompts on the per-language breakdown.
2. **Site shift (scanner memorisation)** - GroupKFold on `site_key`, random-fold AUC reported once as a warning
   number, per-series robust windowing, no scanner metadata as input, crop/rotation augmentations.
3. **Laterality / medial-lateral confusion** - no flips in the baseline; §2.3 experiments are gated on OOF
   medial+lateral AUC; `side_conf` stored in the cache index so canonicalisation can be applied selectively.
4. **T4 speed / fp16** - fp16 autocast verified against fp32 in stage 4; convnext/effnet chosen for fp16
   friendliness; per-study timing measured on the 3 public studies and the 200-study rehearsal; members trimmed
   to 6 h projected; largest studies first.
5. **Offline weight-loading failure** - fingerprint assert in `load_bundle`, `pretrained=False` everywhere,
   wheels installed `--no-index` and tested in a fresh venv, the notebook is generated (no hand edits), export
   verified end-to-end on the 3 public studies (stage 10 `--verify`).
6. **Shared-GPU contention / OOM** - `CUDA_VISIBLE_DEVICES` explicit, `expandable_segments:True`, checkpoint
   every epoch with `--resume`, batch size in config, vLLM `gpu_memory_utilization` capped, `nvidia-smi` check in
   `run_screen.sh` before launch.
7. **Disk** - cache size asserted after build; `verify_cache` reports totals; no second full cache without deleting
   v1; checkpoints keep only EMA weights and the best epoch; `HF_HOME` on /raid, `pip --no-cache-dir`.
8. **9 h limit / long-tail studies** - decode overlapped with GPU, 320-slice 3D series are subsampled before
   resize, studies sorted by size, a wall-clock guard writes the prior-based submission for any study not finished
   by 8 h 15 min so a valid file always exists.

Open uncertainties (honest): vLLM + AWQ throughput on an A6000 (*est.* 30-45 min/variant, could be 2×), the
T4 img/s figure for convnext_small @384 (*est.* 200), whether the hidden test contains compressed transfer
syntaxes (train has none, A§3), the kaggle CLI token format for the new API tokens ([?] in §2.8), and how well
report-derived severity matches the annotators' thresholds (A§Unknowns) - the gold-58 gate is our only handle.
