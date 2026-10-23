# 2. Our pipeline (Szymon): reports → labels → models → Kaggle

The code is in `src/rsna_kad/` and `scripts/`. `pipelineA.py` is a single-file reproduction of the shipped
main_v5_C recipe. The design record is `docs/architecture.md` §1-3.

Note: the local mirror is partly behind the server copy (`/raid/sroziewski/sandbox/rsna_kad`). For example,
`infer.py` imports `FAT_COL`, which the local `series.py` does not define yet. Where they differ, the server copy is
the reference.

## 2.1 Flow

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

## 2.2 Label extraction (`src/rsna_kad/labels/`)

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

## 2.3 Targets (`labels/targets.py`, `labels/pseudo.py`)

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

## 2.4 Folds (`cv.py`)

5 folds, grouped by `site_key` and stratified by language, built with the `balanced` method (best of 200 seeded
greedy assignments, seed 2026). The md5 of `data/folds/folds_v1.csv` is pinned in `tests/test_cv.py`. Gold rows get
a fold but are excluded from every training set (asserted in `train.py`).

## 2.5 Preprocessing cache (`dicom.py`, `series.py`, `preprocess.py`, `cache.py`)

- Per series: sort by InstanceNumber → apply rescale → fill unreadable slices from their neighbours → 130 mm centre
  crop → average 3D slabs to ~3 mm → take ≤ 40 slices → resize to 384 px → p1/p99 window → uint8 `.npy`.
- **Slots** come from (plane, fluid-sensitive): sag_fs, cor_fs, ax_fs, sag_t1, cor_t1, ax_t1, with an FS → T1
  fallback.
- **Laterality** (`laterality.py`) is estimated from the DICOM tag, the series description and geometry, and stored,
  but the models never flip or canonicalise the image (measured gain < 0.001).
- 24,386 series, 98.2 GiB, built in 265 s on 24 workers.

## 2.6 Model (`model.py`, `train.py`)

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

## 2.7 Kaggle export and inference (`kaggle_export.py`, `infer.py`)

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

## 2.8 Infrastructure

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
