# Stage 0-1 QA report - preprocessing cache v1 + folds v1 (tester, 2026-09-18)

Scope: architecture.md §4 rows 0-1 and §3.2 contracts. Everything below was measured on the server
(`/raid/sroziewski/cache/rsna_kad/v1`, params_sha256 65e16190...); scripts and raw outputs are in
`qa/stage1/` (`qa_index.py` -> `qa_index_out.json`, `qa_followup.py`, `qa_montage.py` -> `qa_montage_out.json`
+ `png/`, `qa_bigfov.py`, `qa_centroid.py` -> `centroid.csv`). Server code == local code (md5 of all 20 files identical).

## 1. Developer tests (server)
`TMPDIR=.test_tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_{dicom,series,preprocess,laterality,cache,cv}.py -q -p no:cacheprovider`
-> **47 passed in 17.95 s, 0 failed** (dicom 10, series 6, preprocess 10, laterality 7, cache 9, cv 5). Synthetic fixture < 5 s (asserted). **PASS**

## 2. Cache contract (index.parquet, manifest.json, files)
| Item | Measured | Verdict |
|---|---|---|
| Rows / ok / errors | 24,386 rows (train 24,371 + test 15), ok 24,386, errors 0; 4,410 studies; 0 duplicate series; train series set == train_series.csv; plane/fs == CSV flags; plane-from-IOP == CSV in 24,386/24,386 | PASS |
| Columns vs §3.2 | all 26 contract columns present, dtypes sane (`side_conf` float32); 9 extras: manufacturer, series_number, rows, cols, centre_x, mean_u8, z_step_cached_mm, tag_conf, plane_iop (declared in cache.py) | PASS |
| N range | n_cached 11..40; 0 rows > 40 or < 1; px 384 and crop_mm 130 everywhere | PASS |
| 300 random .npy | 300/300 uint8, ndim 3, (N,384,384), N == n_cached, sidecar present; file mean == index mean_u8 (max abs diff 0.0) | PASS |
| Size | `du -sb` 105,444,831,166 B = 98.2 GiB (105.4 GB); sum of expected npy bytes 105,422,070,016; 24,386 .npy, 0 `*.tmp*` leftovers | PASS (90-145 GB) |
| Build time | logs/cache_20260918-103326.log: train 24,371 series in 261 s (95 series/s, 24 workers), test 3 s, total 265 s, 0 errors | PASS (< 60 min) |
| manifest.json | version v1, params == configs/cache_v1.yaml, params_sha256, n_series 24,386, n_ok 24,386, splits, geom_rules, built_at | PASS |
| Truncated files | both series from pixel_size_mismatch.csv (...28348: n_src 36, ...15703: n_src 38) have truncated_slices=1, ok=True, n_cached==n_src; total truncated 2, no other series flagged | PASS |
| Mean intensity | mean_u8 in [20,200] for 24,377/24,386 = **99.963 %** (median 80.5); 9 outliers: 6 Philips ax_t1 300 mm FOV (mean 11-19, see finding F4), 2 Philips ax_fs with FOV 70/81 mm (padding), 1 GE sag_t1 at 202 | PASS (>= 99 %) |
| p_lo / p_hi | p_hi > p_lo in all rows; p_lo median 1.3 (min -18.7, 5 negative = Toshiba), p_hi median 981 (89 .. 116,859 - raw vendor units, expected); p_lo == 0 in 9,947 rows (41 %: Philips 6,626, GE 2,011 - zero-filled backgrounds), 104 of the 106 padded (FOV < 130) series have p_lo = 0 | PASS |
| Slot counts (series / studies>=1) vs data_analysis §2 | sag_t1 5,197/4,266 - ax_fs 4,719/4,407 - sag_fs 4,667/4,150 - cor_fs 4,624/4,248 - cor_t1 3,985/3,406 - ax_t1 1,179/857: all six pairs identical to the A§2 table; exactly one `selected` per (study, slot); rank-0 never a 3D series when a 2D sibling exists (0 cases) | PASS |
| Fallbacks | studies without sag_fs 257, without cor_fs 159, without ax_fs 0; `assign_slots` fallback rows: sag_fs<-sag_t1 257, cor_fs<-cor_t1 159 | PASS |
| 3D series | 837 (ax_fs 318, sag_t1 269, sag_fs 148, ax_t1 101, cor_fs 1); n_src 50-320; z 0.4-1.6 mm -> cached z 2.4-6.09 mm (median 3.0, p75 3.22); n_cached max 40, 43.7 % hit the cap; every 2D series has n_cached == min(n_src, 40) (740 2D series > 40 subsampled); 0 untagged series > 60 slices misclassified | PASS |

## 3. Geometry by eye (12 montages in qa/stage1/png: 8 evenly spaced cached slices, original middle slice with the 130 mm crop box, cached middle, my independent recompute, |diff|)
Independent re-derivation (own code: InstanceNumber sort, slope/intercept, mm crop, 3 mm slabs, cap 40, resize, p1/p99): **max |diff| = 1 grey level, mean 0.00 on all 12 cases and on 60 random series (45 2D + 15 3D); InstanceNumber order monotone along IPP in 72/72**.
- 01 Siemens sag FS 512^2, 0.332 mm, FOV 170: box visibly smaller than the original, knee centred, lateral->medial continuous, 1.0 % at 255 / 1.3 % at 0. OK.
- 02 Siemens cor FS 640x540 (non-square), 0.25 mm: no aspect error, joint centred, post->ant continuous; 7.9 % zeros are the scanner's zero bands. OK.
- 03 Siemens ax FS 384^2, 0.52 mm, FOV 200: crop 250 px upscaled x1.54 (INTER_LINEAR), slightly soft, sup->inf continuous. OK.
- 04 Siemens DESS 3D 640x1280, 144 x 0.6 mm -> 29 slabs at 3.0 mm: wide black bands removed by the mm-crop, no distortion, slabs clean and continuous. OK.
- 05 GE 1024^2, 0.156 mm, 52 slices -> 40 (z 2.2->2.88): crisp INTER_AREA downscale, continuous; 13.8 % zeros = GE masked background. OK.
- 06 GE 256^2, 0.74 mm, FOV 190: x2.2 upscale, blurry as the source dictates, centred. OK.
- 07 Philips 704^2, RescaleSlope 4.87 (per-file slopes): window sensible (p 0/1580), no z-flicker. OK.
- 08 Toshiba signed int16 640x760 (raw min -454): negatives are noise below p1=0, T1 fat not clipped (1 % at 255). OK.
- 09 truncated series ...28348 (ax FS 36 slices): corrupt file is the LAST slice (z=35) -> filled as a copy of z=34 (edge case of neighbour fill), identical to my recompute; rest normal. OK.
- 10 GE cor T1 FOV 120 mm: ~15 px zero frame around the image, scale preserved. OK.
- 11 GE cor FS 256^2, FOV 200: joint inside the box, x2.3 upscale. OK.
- 12 GE sag T1 512^2, FOV 170: centred, continuous, window fine. OK.
- 13 (extra, `13_bigfov.png`): 85 series have FOV >= 230 mm. GE 240 mm ax/sag: knee inside the crop. **Philips ax_t1 300 mm 256^2: knee sits in the lower half of the field, the centre crop contains mostly air** -> finding F4.
Whole-cache sweep (`qa_centroid.py`, middle slice of all 24,386 series): centroid offset from centre median 20 px (6.8 mm), p90 38 px, p99 56 px; > 90 px (30 mm) in 13 series, all Philips ax_t1 300 mm; centre-patch mean < 15 in 6 series (same group). **PASS for all main slots; ax_t1 caveat.**

## 4. Laterality (recomputed from the index with the developer's aggregation: axial > coronal > sagittal centre_x)
- side_geom vs side_tag on tagged studies: **Siemens 99.03 % (n 1,644), Philips 98.52 % on the 473 consistently-tagged (97.59 % on all 498 incl. the 25 contradictory), GE 62.16 % (n 37), Hitachi/Fuji 62.5 % (n 24), Canon: no tagged study -> no rule.** Matches the developer's claim (99.0 / 97.6 / 62) and manifest geom_rules; trusted = {siemens, philips} only. >= 95 % criterion met for Siemens/Philips. PASS
- side_est source (4,407 train studies): tag 2,203, desc 289 (conf 0.8), geometry 1,025 (Philips 766 @0.985, Siemens 259 @0.990), unknown 890 (GE 664, Canon 226). **Geometry never feeds side_est for GE/Hitachi/Canon** (0 cases); `side_geom` column is still filled for GE/Hitachi from the untrusted rule (905+24 rows), informational only, fine. PASS
- Contradictory tags: 25 studies (all Philips), tag_conf 0.6 -> side_conf 0.6 in 25/25; tag wins over desc in the 4 tag/desc conflicts. PASS
- Cross-check: untagged studies with both a description side and a geometric side: Philips 37 + Siemens 48 agree 100 %; GE 204 agree **0 %** (i.e. those GE sites are perfectly consistent with the opposite sign: the GE convention is site-specific, not random; see F7).
- Test split: Siemens study L (tag), Toshiba unknown, GE unknown (side_geom "L" ignored). Consistent.

## 5. Folds (data/folds/folds_v1.csv)
Columns `StudyInstanceUID,fold,site_key,lang,is_gold`; 4,407 rows, ids unique == train.csv == cache train studies; folds 0-4; is_gold True 58; 123 site_keys, **0 sites in two folds**; sizes 19.95 / 19.95 / 20.04 / 20.01 / 20.04 %; gold 11 / 11 / 12 / 12 / 12; site_key/lang/labelled identical to study_summary.csv; `cv.check_folds` = []; **md5 4dc54b74429f07051af2175e7bb35442 == FOLDS_V1_MD5 in tests/test_cv.py**; regeneration with configs/folds_v1.yaml (balanced, seed 2026, 200 trials) reproduces the file exactly. **PASS on every §4 criterion.**
Language shares per fold vs global: max deviation **23.8 pp** (de: 262 studies = 1 site -> all in fold 0; bg 220 = 1 site -> fold 3; el 2 sites -> folds 2/4; nl 3 sites; fr 2 sites). The +-3 pp target in my brief is infeasible under site grouping (sklearn StratifiedGroupKFold: 36 pp, sizes 14-25 %, gold 9-17); en/es/tr deviate <= 1.3 / 1.3 / 7.9 pp. Not a defect, but per-language OOF for de/bg/nl/fr will come from 1-2 folds, and site-shift is exactly what the split measures. Deviation from the text of §2.5 (which says StratifiedGroupKFold): documented in cv.py/config; architecture.md should be updated.

## 6. Resume / idempotency
`python scripts/build_cache.py build --split test --workers 2` -> "15/15 done (15 skipped, 0 errors)", 1 s. All 30 files under `v1/test/` (npy + sidecar json): size and mtime unchanged (`find -printf` diff empty), npy md5 unchanged. `index.parquet` and `manifest.json` are rewritten on every build call (by design): content identical (`DataFrame.equals` True, manifest equal except `built_at` 10:40:24 -> 14:35:14). Note: this run therefore touched those two files' timestamps; nothing else in the cache was modified. **PASS**

## 7. Code review (correctness only): checked and found correct
Axis order (N,H,W) throughout; InstanceNumber sort with path tie-break, no off-by-one (verified on 72 series vs IPP); RescaleSlope/Intercept applied exactly once in `read_slice` (pydicom applies no LUT); signed int16 preserved (Toshiba p_lo -18.7 in 5 series, negatives windowed not truncated); non-square crop centred per axis with zero padding; INTER_AREA on downscale / INTER_LINEAR on upscale; 3D slab averaging then cap -> N <= 40 always (verified 837/837); `subsample_indices` yields exactly 40 unique indices for n > 40; fallback rows never counted as native; sibling pick only among rank >= 1; test-split build reuses the manifest geom rules (no refit on 3 studies).

### Findings (none blocks acceptance)
- **F1 (low, robustness)** `src/rsna_kad/cache.py:87-91`: `slot_of(task["plane"], task["fs"])` runs *outside* the `try` in `process_series`; an unexpected plane/fs value would raise in the worker and abort the whole `imap_unordered` build (docstring says "never raises"). No impact on this data (3 planes, 0/1 flags); matters for the Kaggle path if the hidden test CSV differs.
- **F2 (low, memory)** `src/rsna_kad/preprocess.py:130-134`: `np.stack` materialises the full-resolution float32 volume *before* `crop_centre` (the comment claims the opposite): a 320x640x1280 series costs ~1.05 GB for the stack plus ~1.05 GB for the decoded list per worker. Fine with 24 workers on the server; keep in mind for `Pool(3)` on Kaggle.
- **F3 (low)** `preprocess.py:143`: p1/p99 are computed on the *padded* volume for FOV < 130 mm (106 series, 0.43 %): p_lo is forced to 0 and p_hi drops to ~p96-p99 of the real voxels (worst case FOV 70 mm = 71 % padding). Deterministic and reproduced at inference; a v2 could mask the pad.
- **F4 (low for main arms, medium if ax_t1 is ever used)** fixed centre crop vs off-centre anatomy: 13-19 Philips `ax_t1` series with 300 mm FOV have the knee in the lower half of the field; the 130 mm centre crop holds mostly air (6 with centre mean < 15, mean_u8 11-19). Slot `ax_t1` is ignored by MAIN_SLOTS, so no effect on stages 3-8; blacklist or centroid-centre these before any 6-slot arm.
- **F5 (low, stage 4)** `src/rsna_kad/dicom.py:117`: `read_slice` catches `(ValueError, OSError, AttributeError, InvalidDicomError)`; pydicom 3 raises `RuntimeError`/`NotImplementedError` when no decoder plugin handles a compressed transfer syntax -> whole-series error instead of per-slice fill. Acceptable in the cache (ok=False row), but the inference preprocessor must catch it per slice as §2.7 promises.
- **F6 (doc)** folds are produced by the seeded greedy "balanced" method, not `StratifiedGroupKFold` as written in architecture §2.5; the deviation is justified (sklearn gives 14-25 % folds) and pinned by md5, but the architecture text is now wrong.
- **F7 (opportunity, stage 9)** `laterality.fit_geom_convention` fits one sign per manufacturer *family*; GE is 62 % because its sites disagree with each other (204 GE studies with a description side contradict the fitted sign 100 %). Fitting per `site_key` with tag+desc as supervision would likely recover most of the 664 unknown GE studies.
- Edge-truncation note: a truncated first/last slice is filled by copying its single neighbour (spec says neighbour mean); reasonable, and it is the actual case for series ...28348.
- Inference-parity: `cache.py` uses nothing the notebook cannot reproduce (pydicom + numpy + cv2 only; CSV flags for slot; manifest `geom_rules` must be shipped in the bundle for `side_geom`). cv2/numpy version drift can move single grey levels (my recompute already shows +-1), so stage 4's 99.9 % uint8-equality gate is the right test.

## Verdict
All §4 stage-0/1 acceptance items pass with measured numbers (tests 47/47; 24,386 rows all ok; 98.2 GiB; 265 s build; shapes/dtypes on 300 files; 99.96 % mean in range; 2 truncated flagged; folds 5 x ~20 %, gold 11-12, 0 site leaks, md5 pinned; Siemens 99.0 % / Philips 98.5 % geometry agreement; resume skips 15/15 without touching data files). Independent recomputation matches the cache to +-1 grey level on 72 series and all 12 visual cases are anatomically correct. Findings F1-F7 are non-blocking.

**stage 1 ACCEPTED** (with F1-F5 recommended for the stage-4 inference path and F4 to be handled before any ax_t1 use).