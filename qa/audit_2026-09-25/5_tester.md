# Audit 2026-09-25 - TESTER report (independent verification by execution)

Server `/raid/sroziewski/sandbox/rsna_kad`, CPU only (`CUDA_VISIBLE_DEVICES=`, <= 8 procs, `HOME=$PWD/.home`), nothing
written outside `qa/audit_2026-09-25/`. No Kaggle submission, no push, no dataset version. Token never printed.
Evidence: `qa/audit_2026-09-25/tester/` (local copy of the logs + drill scripts `drill_edge.py`, `recompute_armc.py`).
Code parity local vs server: every `src/ scripts/ tests/ configs/` file md5-identical; **only `notebooks/kaggle_inference.py` differs** (B1).

## Verdict per item
| # | Item | Result |
|---|---|---|
| 1 | Full test suite | **PASS** - 275 passed, 0 failed, 0 skipped, 58.3 s (1:00 wall) |
| 2 | Live submission = what we think; published pair coherent; kernel reproduces locally | **PASS** (diff 1.78e-4) |
| 3 | Runtime safety / wall guard / compressed DICOM | **PASS with caveats** (runtime not directly measured, B2-B4, B7) |
| 4 | Robustness drills (13 edge studies) | **PASS** - 13/13 valid rows |
| 5 | Reproducibility (targets v1, arm C metrics) | **PASS** - bit-identical / diff 0.0 |
| 6 | Data integrity (truncated DICOMs, cache, folds md5) | **PASS** |

## 1. Test suite
`CUDA_VISIBLE_DEVICES= TMPDIR=$PWD/.test_tmp PYTHONDONTWRITEBYTECODE=1 /raid/sroziewski/sandbox/bin/python -m pytest tests -q -p no:cacheprovider -o addopts="" -rs --durations=15`
(+ `OMP_NUM_THREADS=4`) -> **`275 passed, 21 warnings in 58.26s`**, exit 0, peak RSS 1.45 GB. Architecture §4.1 last recorded 258.
Includes `test_committed_notebook_equals_fresh_build` (server notebook == build from current src) and
`test_wall_clock_guard_returns_every_study`. Warnings: 15x pin_memory-without-GPU (test_train), 1x
`DeprecationWarning: multi-threaded ... fork() may lead to deadlocks` (test_infer; see B2).

## 2. The live submission
- `kaggle_cli.sh submissions`: **56488491** COMPLETE, public **0.933**, 2026-09-23 08:40:58, description "main_v3 ... 15 members
  ... dataset v5, kernel v10"; API field `url = /code/sroziews/rsna-kad-mvp?scriptVersionId=352079831`; **total_bytes 222,263**.
  A 3-row file is 650 B; at uid+109 B/row that is **~1,284-1,299 rows** = a full hidden-test file (scored on the whole hidden set).
- Kernel (kagglesdk `get_kernel`): `current_version_number = 10`, T4 machine, internet off, source `sroziews/rsna-kad-weights`;
  `kernels status` COMPLETE. Dataset (`get_dataset`): `current_version_number = 5`, created 2026-09-23 08:34:13, notes "main_v3 = ...".
  **Nothing has been pushed or versioned since the submission** -> the live pair is exactly v10 + v5.
- Kernel code (`kernels pull`) md5 **e61dc63c61db3a3324e125d6fc7c66e4 == server `notebooks/kaggle_inference.py`**.
- Dataset FILE LISTING (`datasets files --page-size 200`, 41 files) == `export/main_v3` names + byte sizes exactly (only
  `dataset-metadata.json` absent, expected). Downloaded 23 files (VERSION, ensemble.json, all 15 member jsons, f0+f4 safetensors of
  all 3 arms): **23/23 md5-identical** to `export/main_v3`.
- Kernel v10 log: wheels installed offline (5 packages, cp312), 15 members loaded with fingerprint check, 3 studies, 0 prior rows.
- Local re-run today (committed notebook, `export/main_v3`, CPU fp32, 3 workers): 1,328 s (442.7 s/study), peak RSS 10.1 GB.
  **kernel vs local max|diff| = 1.78e-4** (precedent 2.1e-4); today's run == the 2026-09-23 local run **bit-exact (0.0)**;
  ids and header order identical.
- **Would versioning the dataset / pushing the kernel now break the selected entry?** No for its score: the private score is computed
  from the 222 KB file already produced by the hidden rerun of v10; selection does not re-run anything. Residual risk only if Kaggle
  ever re-runs submissions (host rescoring) - then v10 must still find a compatible `rsna-kad-weights`. Recommendation: publish any
  future bundle as a **new dataset slug** (or never delete v5) and keep kernel v10 untouched; per RELEASE HAZARD, any *new*
  submission still needs dataset+kernel released together and kernel-vs-local verified. The pair is coherent for final selection.

## 3. Runtime safety (main_v3, hidden ~1,290 studies, 9 h limit)
- `logs/benchv3_20260923-101845.log`: one A6000, 15 members, 24 studies: **6.14 s/study** (prep 1.17, GPU 5.61), 0 errors;
  bench's own projections: 133 min A6000, **598 min "T4" / 748 min with margin** - that convention assumes ONE T4 (x4.5).
- Kaggle uses 2 T4s, but `assign_devices` is round-robin by count: cuda:0 holds 8/15 members (3 A, 2 E, 3 C) ~ **58 % of GPU cost**
  (estimated from bench main_v2 2.92 s vs main_v3 6.14 s). Projection: empirical factor 2.2 (MVP bench 0.53 s -> hidden 25 min)
  -> **~4.9 h**; conservative x4.5/T4 with the 58 % split -> **~5.8 h** (x1.25 = 7.2 h). Margin to 9 h: **~3.2-4.1 h**; to the
  8.25 h guard ~2.5-3.4 h. Actual hidden runtime is **not observable via the CLI** (check the submission page in the UI); the run
  finished (COMPLETE, full-size file) and LB 0.933 > 0.929 makes a guard-fired run (prior rows) implausible.
- `logs/localv3_...`: CPU 273 s/study (09-23); `logs/pushv3_...`: dataset upload OK 10:30-10:34.
- **Wall guard works**: `ensemble.json` `wall_guard_hours 8.25`; test `tests/test_infer.py:305` passes. Drill with the real notebook
  (`--wall-hours 0`, 3-member mini export, 3 public studies): "wall-clock guard hit after 1/3 studies" -> CSV 3 rows, 1 predicted,
  **2 exact prior rows**, all in [0,1], exit 0. Code path (infer.py:529-561): stop flag, pool terminated, unfinished -> prior row,
  `write_submission` validates. Caveats B2/B3.
- **Compressed DICOM**: no test encodes a real compressed file (only mocked RuntimeError/NotImplementedError, test_infer.py:112-135).
  Drill: wheels from `export/main_v3/wheels` installed `--no-index` (all 5 OK); the smallest public test study transcoded with gdcm to
  **JPEG-LS lossless, JPEG2000 lossless, JPEG lossless P14 SV1, RLE** -> `StudyPreprocessor` slot volumes **identical (max|diff| 0)**
  to the uncompressed original for all 4 syntaxes, with and without gdcm/rle present; end-to-end rows identical (mean_p 0.3132 x4).

## 4. Robustness drills (`drill_edge.py predict`; conftest `write_series`; 3-member mini export A/E/C f0; CPU fp32)
| case | n_slots | row | note |
|---|---|---|---|
| one series only (sag_fs) | 1 | valid, predicted | |
| missing coronal plane | 2 | valid, predicted | |
| 3D-only (sag 120 x 0.6 mm, cor 100 x 0.7 mm) | 2 | valid, predicted | |
| truncated slices (first, middle, last) | 2 | valid, predicted | neighbour fill |
| non-square 640x1280 and 1280x640 | 2 | valid, predicted | |
| every slice truncated | 0 | valid, prior | "no usable slot" |
| unknown plane "Oblique" in CSV | 1 | valid, predicted | bad series skipped (F1 fixed) |
| dir missing / study absent from test_series.csv | 0 | valid, prior | |
| 4 compressed transcodes of a real study | 4 | valid, predicted | identical |
13/13 rows, header/order == sample_submission, all finite in [0,1], exit 0.

## 5. Reproducibility
- `scripts/make_targets.py --labels data/labels/llm_v1.parquet --config configs/targets_v1.yaml --folds data/folds/folds_v1.csv
  --out qa/.../targets_check` -> **11/11 files md5-identical** to `data/targets/v1` (5 parquet, 5 calibration json, manifest).
- Arm C (`runs/main_armc_convnextv2_t_384`, last write 2026-09-22 05:29, no live screen on it): oof == concat(fold oofs), 4,407 rows,
  fold column == folds_v1. Independent sklearn: **weak 0.925352 / strict 0.857635 / gold-58 0.887812 = metrics.json, diff 0.0**.

## 6. Data integrity
cache v1 `index.parquet`: **24,386 rows, 24,386 ok, 0 errors** (train 24,371 / test 15), 0 missing .npy, manifest n_ok 24,386.
The 2 truncated series (...28348 36/36, ...15703 38/38) have `truncated_slices=1, ok=True`; re-decoded through the *inference*
path they are **identical to the cache**. `folds_v1.csv` md5 **4dc54b74429f07051af2175e7bb35442** == `tests/test_cv.py:12`.

## Bugs / findings
| id | where | scenario | severity |
|---|---|---|---|
| B1 | local `notebooks/kaggle_inference.py` (md5 0ed40f4f, 09-21 18:27) vs server/kernel e61dc63c | local copy is stale (not rebuilt after the 09-23 src change); `scripts/remote.ps1:138` `sync` pushes `notebooks` and would overwrite the deployed-equal server copy; local `test_committed_notebook_equals_fresh_build` fails | medium (process); functional impact nil for main_v3 - stale vs current notebook on the same study: max diff 0.0 |
| B2 | `src/rsna_kad/infer.py:516` `pending.popleft().get()` (no timeout); pool is `fork` after torch threads exist (pytest DeprecationWarning) | a hung decode or an OOM-killed Pool worker blocks forever, so the 8.25 h guard never runs -> notebook timeout, no submission | low (hidden set already passed once) |
| B3 | `infer.py:409-423, 543-545` | any exception in a consumer for one study (e.g. CUDA OOM) aborts the whole run instead of a prior row | low (fixed input shapes) |
| B4 | `infer.py:301-305` `assign_devices` | round-robin by count: cuda:0 carries ~58 % of GPU cost -> ~15 % longer wall than a cost-balanced split | low (perf) |
| B5 | tests | no test decodes a real compressed DICOM; add an RLE (pydicom-native) or J2K round-trip test like this drill | low (coverage) |
| B6 | `docs/architecture.md` §4.1 | main_v3 / submission 56488491 / dataset v5 / kernel v10 / LB 0.933 not logged; suite count says 258 | doc |
| B7 | `scripts/bench_infer.py:234` | `passed` = per-study < 60 s, not the §2.7 budget; main_v3 reports passed=true with 748 min single-T4 projection; 2-GPU split not modelled | low (tooling) |

## Final verdict
**The current best entry (56488491, main_v3, kernel v10 + dataset v5, public 0.933) is SAFE to leave as a final selection.**
It is already scored on the full hidden set; the published kernel and dataset are exactly the committed notebook and `export/main_v3`;
a local run reproduces it to 1.8e-4; the guard, decoders and edge-case paths all produce valid rows. Before the deadline: select it
explicitly in the UI, fix B1 (rebuild/pull the local notebook), and ship any future bundle under a new dataset slug.
