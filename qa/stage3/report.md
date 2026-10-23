# QA stage 3 - model + MVP training (tester report, 2026-09-18)

Scope: architecture §2.4, §2.5, §3.2 (OOF/bundle), §4 row 3. Read-only w.r.t. src/scripts/tests/configs/runs.
Evidence scripts + raw outputs: `qa/stage3/audit_leak_oof.{py,out}`, `qa/stage3/audit_bundle_data_model.{py,out}`
(same paths on the server under /raid/sroziewski/sandbox/rsna_kad). All server work on CPU (`CUDA_VISIBLE_DEVICES=""`),
4 threads, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=$PWD/.test_tmp`. Run audited: `runs/mvp_effv2s_256` (5 folds finished).

## 1. Developer tests - PASS
`python -m pytest tests/test_data.py tests/test_model.py tests/test_train.py -q -p no:cacheprovider` -> **19 passed, 0 failed**
(test_data 8, test_model 6, test_train 5; only pin_memory-on-CPU warnings).

## 2. Leakage audit - PASS (no leak found)
Reconstructed every fold's split with the code's own `train.split_studies` from `folds_v1.csv`
(md5 `4dc54b74429f07051af2175e7bb35442`, 4,407 rows) + `data/targets/v0/fold{k}.parquet` + cache index (4,407 cached train studies).
Gold id sets agree: `gold_ids.txt` = `folds.is_gold` = `train.csv` non-NaN rows = 58.

| fold | train (log / rebuilt) | val (weak+gold) | train ∩ gold | train ∩ fold_k | oof rows == fold_k ids | gold rows in targets |
|---|---|---|---|---|---|---|
| 0 | 3480 / 3480 | 880 (11 gold) | 0 | 0 | True (880) | 0 |
| 1 | 3481 / 3481 | 879 (11) | 0 | 0 | True (879) | 0 |
| 2 | 3478 / 3478 | 883 (12) | 0 | 0 | True (883) | 0 |
| 3 | 3479 / 3479 | 882 (12) | 0 | 0 | True (882) | 0 |
| 4 | 3478 / 3478 | 883 (12) | 0 | 0 | True (883) | 0 |

- (a) `fold0/config.yaml` = MVP config (targets v0, folds_v1); train set = folds!=0 AND not is_gold AND has target AND cached; code also
  asserts `not gold_ids & train_ids` (train.py:262). Zero gold studies in any training set.
- (b) Validation = exactly fold 0 (880 = 869 weak + 11 gold); `fold0/oof.parquet` ids == fold-0 ids, `fold` column == 0.
- (c) Targets v0: 4,349 rows, 0 gold rows, all 5 fold files byte-identical, `calibration_fold{k}.json` has `n_gold_train=0`;
  `configs/targets_v0.yaml` has `calibrate: false` and `targets.calibrate()` returns the base table unchanged in that case.
  **v0 uses no gold calibration at all**, so fold-0 targets cannot derive from fold-0 gold. (v1 must be re-audited at stage 6.)
- (d) Best epoch is chosen on the validation fold (train.py:360, `>=` so ties pick the later epoch). Measured optimism, all 5 folds,
  weak AUC per epoch (best in bold): f0 .769 .828 **.850** .849 .839 .834 / f1 .774 .833 .849 **.852** .852 .850 / f2 .779 .820 .842 **.850** .848 .845 /
  f3 .800 .847 .868 **.872** .869 .865 / f4 .757 .780 .793 **.808** .804 .800. **Mean best 0.8462 vs last-epoch 0.8386 -> +0.0076 optimism**
  in the reported OOF-weak; gold at best-weak epoch 0.865 vs last 0.852 (per-fold means). Val AUC peaks at epoch 2-3 and decays
  while train loss keeps falling (0.62 -> 0.46): late-epoch overfitting to noisy v0 labels. See finding F1.
- (e) No metadata input: grep of data.py/model.py/train.py for site_key/side_*/Manufacturer/SeriesDescription/lang/scanner -> 0 hits.
  The model sees only uint8 slice triplets + 3 presence bits (model.py:96). `CacheIndex` uses index columns plane/fs/rank only.
- (f) No flips: grep `flip|fliplr` over src/ -> 0 hits; `warp_slices` does rotation/zoom/brightness/contrast only.

## 3. OOF / metrics contract - PASS (one metric-definition ambiguity, F2)
`runs/mvp_effv2s_256/oof.parquet`: columns exactly `[StudyInstanceUID, fold, {key}_pred x12]`, 4,407 rows, 4,407 unique ids,
== all folds_v1 ids (58 gold included), float32, range [0.0059, 0.9967], 0 NaN, `fold` agrees with folds_v1 for every row.
`metrics.json` keys: oof_weak_auc, oof_strict_auc, gold_auc, gold_ci95, per_label (+ per_label_strict, per_label_gold, n_weak,
n_gold, n_rows, folds) - superset of §3.2. Bundle metrics epoch (2) == fold-0 best epoch; OOF written at the same epoch.
sklearn recomputation (§2.5: pos >= 0.6; neg <= 0.2 and w >= 0.5; middle dropped; per-fold v0 targets; 4,349 weak rows):
- 11/12 labels identical to 4 decimals (acl .8758, med_men .8800, lat_men .7324, med_oa .8569, lat_oa .8201, pf_oa .8250,
  effusion .8857, synovitis .8374 (n_neg=22!), baker .9178, contusion .8415, fracture .8919).
- **mcl: sklearn 0.7773 vs reported 0.7910.** Cause (verified): mcl `affirmed_mild` base = 0.20 == NEG_THR; the targets are float32
  and `metrics.binarise` upcasts to float64, so 0.2f32 = 0.2000000030 > 0.2 and the 552 mild-MCL rows (w >= 0.5) fall in the
  dropped band (n_neg 3,476) instead of the negatives (n_neg 4,028 with an exact-precision compare).
- Macro: **recomputed 0.8452 vs reported 0.8463 (delta -0.0011, entirely the mcl boundary)**; per fold sklearn/reported:
  f0 0.8484/0.8495, f1 0.8501/0.8517, f2 0.8491/0.8503, f3 0.8708/0.8717, f4 0.8061/0.8078.
- Gold-58: **recomputed 0.8466 == reported 0.8466** (delta 0.0), 58 rows joined, no NaN label; reported CI95 [0.815, 0.877].
  Per-label gold: mcl 0.58 and lat_men 0.73 are the weak ones; acl/med_oa/effusion >= 0.94.
Acceptance §4 row 3: OOF-weak >= 0.75 -> 0.8495 fold 0 (0.8452-0.8463 5-fold) PASS; gold reported PASS.

## 4. Bundle contract - PASS
`fold0/bundle.json` keys == {run_id, fold, arch, norm, labels, fingerprint, config, metrics, git_sha} (no missing/extra);
arch = {K_infer, attn_hidden, backbone, crop_mm, fusion, img_px, slots}; norm = {mean, std, p_lo, p_hi}; 12 labels; 8 fingerprints;
`git_sha` = "" (repo not under git on the server, `RSNA_KAD_GIT_SHA` unset - informational).
`model.safetensors` = 92,128,844 bytes (x5 folds, identical size), 793 tensors, 22,857,857 params (backbone 20,177,488).
`load_bundle(dir, "cpu")` 0.3 s, model in eval mode; **bit-exact reload** (all 793 tensors `torch.equal` to the raw file).
Tamper tests on a copy under qa/stage3/tamper (deleted afterwards):
- A: zero a fingerprinted tensor (`backbone.blocks.4.2.bn2.running_var`) -> `RuntimeError: weight fingerprint mismatch` PASS.
- C: remove one tensor -> `RuntimeError: missing [...]` (strict load) PASS - this is the stage-4 acceptance case.
- B: zero a non-fingerprinted tensor (`backbone.blocks.5.5.bn1.running_var`) -> **not detected** (by design: 8-tensor fingerprint,
  §2.7). Note for stage 4: fingerprint catches wrong/random init, not arbitrary partial corruption.
CPU fp32 re-prediction of 8 fold-0 studies from the bundle vs oof.parquet (bf16 GPU): max |dp| 0.0069, mean 0.0017.

## 5. Data pipeline (real cache v1, CPU) - PASS
- (a) `sample_centres`/`triplet_indices`: N=11,K=24 uniform -> [0,0,1,1,2,2,2,3,3,4,4,5,5,6,6,7,7,8,8,8,9,9,10,10]; N=30 ->
  [0,1,3,...,28,29]; N=40 -> [0,2,4,...,37,39]; first/last triplets edge-clamped [[0,0,1],[n-2,n-1,n-1]]; all indices in [0,n-1].
  K=16 stratified-random: every slice hit in 200 draws (min hits 184/76/65 for N=11/30/40), no slice unreachable.
- (b) `model.normalize(uint8)` == (x/255 - mean)/std (allclose 1e-6), float input passes through; the same function is used in
  `KneeMIL.embed` for train and inference, dataset returns uint8 -> one normalisation path. PASS.
- (c) Eval item == plain `cv2.INTER_AREA` resize of the cached slices (array_equal) -> no augmentation at train=False; with params
  `read_triplets` == whole-volume `warp_slices` gathered at the triplet indices -> identical params across all slices of a series
  (params drawn once per slot per item, data.py:158). Train item differs from eval item (aug active).
- (d) `slot_dropout_p=1.0, sibling_p=1.0` on 300 studies: natural presence [3]x300 -> after dropout [2]x300; min present = 2;
  never < 2 (guard `present.sum() > 2`, data.py:149). 1,419 (study,slot) pairs have >1 sibling series, so sibling swap is live.
- (e) MVP slots sag_fs/cor_fs/ax_fs: 0 studies missing any slot after fallback (so the path is exercised only synthetically):
  synthetic batch with slot 1 absent -> embed row is zero, `fill` row == `missing[1]`, present rows untouched; perturbing
  `missing[1]` changes only the study with the absent slot; forward (2,12)+(2,3,12) finite. PASS (acceptance item).
- (f) Eval K = 24, train K = 16; eval item deterministic across calls, rng None -> uniform strata. PASS.
- (g) Same seed/epoch -> identical items (two dataset instances); epoch change -> different; DataLoader with 2 workers and a
  seeded generator reproduces the same study order and images. PASS.

## 6. Model - PASS
22.86 M params; `train.log` line 2: "Loading pretrained weights from Hugging Face hub (timm/tf_efficientnetv2_s.in21k_ft_in1k)"
(safetensors, HF_HOME cache); corr(timm pretrained conv_stem, bundle EMA conv_stem) = **1.0000** (max |d| 0.016) vs 0.032 for a
random init; blocks.5.0.conv_pw corr 0.999 -> the fine-tuned weights descend from the ImageNet checkpoint. Gated attention:
softmax over K, sums to 1 per slot (5x16 check). Aux heads present: `aux_w` (3,1280,12), `aux_b` (3,12); `missing` (3,1280).

## 7. Training loop review (read-only) - PASS with notes
- EMA: evaluated (`predict_studies_cached(ema.module, ...)`, train.py:348) and saved (`save_bundle(ema.module.state_dict())`, :363)
  weights are the EMA; covers BN running stats. Deviation from §2.4 "0.999/step": decay ramps as min(0.999,(1+t)/(10+t)) -> with
  5,220 steps the cap is never reached (0.9983 at the end); documented in the docstring, sensible for 6-epoch runs. (note N1)
- LR groups: backbone 1e-4 / heads 1e-3 (`head_parameters` = all non-backbone params); warmup verified in the log
  (step 50/870: lr 5.86e-6 = 51/870*1e-4), cosine over epochs*steps. Weight decay 0.05 hits biases/BN/`missing` too (N2, low).
- Grad clip 2.0 on all params before `opt.step`; bf16 autocast, loss in fp32 (`logits.float()`); no GradScaler, so `amp_dtype:
  float16` would be unsafe (N3, not used by any config).
- `masked_bce` = sum(bce*w)/sum(w) verified numerically; targets v0 have no weight-0 cells (min weight 0.1).
- Manifold mixup: same `lam`/`perm` applied to filled embeddings, presence bits, targets and weights; aux mask = mixed weight x
  mixed presence - consistent. Design note (N4): plain linear mixing lets a w=0.1 cell (t=0.15) pull a w=1.0 affirmed cell to
  0.535; a weight-aware mix would be more principled, but §2.4 specifies linear mixing.
- Resume: restores model/EMA/opt/sched/history/best/epoch from last.pt, mix_rng reseeded with start_epoch; the loader shuffle
  generator is not advanced, so post-resume epoch order repeats epoch-0 order (cosmetic, N5). No config-equality check on resume.
- Val fold isolation: only `split_studies` builds the lists; train dataset indexes train_ids only; val evaluation uses the same
  target table but inner-joins on predicted ids. Nothing trains on fold k. Peak mem 2.7 GB, ~470 img/s, 40 min/fold.

## 8. Live run check - PASS
`screen -ls`: no `mvp` session (finished); `logs/mvp_20260918-144200.log` ends `exit: 0 at 2026-09-18T18:04:37`. All 5 folds done:
best epoch/weak/strict/gold: f0 2/0.8495/0.810/0.846 / f1 3/0.8517/0.811/0.848 / f2 3/0.8503/0.798/0.888 / f3 3/0.8717/0.837/0.863 /
f4 3/0.8078/0.773/0.879; wall 40 min each (fold 0: 14:42-15:22, < 1 h acceptance). Smoke run (`logs/smoke_20260918-143453.log`):
200 train studies, 6.7 min, weak 0.670 > 0.55 (acceptance). nvidia-smi at audit time: GPU0 3.8 GB, GPU1 10.8 GB used by other
users' processes, none ours.

## Findings
| id | where | what | severity |
|---|---|---|---|
| F1 | src/rsna_kad/train.py:360 | Best-epoch selection on the validation fold makes oof_weak_auc optimistic by +0.0076 (mean over 5 folds) and the bundle/OOF epoch data-dependent. Report the last-epoch OOF alongside, or fix the epoch count from the MVP before stage 6 (main arms, 12 epochs) so OOF stays a clean estimate. | medium (reporting bias, not leakage) |
| F2 | src/rsna_kad/metrics.py:44 + configs/targets_v0.yaml mcl affirmed_mild=0.20 | Negative-band membership of exactly-0.2 targets depends on float32->float64 upcast (mild MCL dropped only because 0.2f32 > 0.2). Macro effect +0.0011, mcl +0.0137. Make the intent explicit (strict `<` or move the base off the threshold) and pin it in a test. | low |
| N1-N5 | train.py:123, :103, :316, :319-325, :273 | See §7 notes (EMA ramp deviation, wd on biases/BN, fp16 without scaler, linear mixup of weights, shuffle order after resume). | low / informational |
| B | model.py:111-115 | Fingerprint covers 8 of 793 tensors; a zeroed non-fingerprinted tensor loads silently. Acceptable per §2.7 (wrong/random init is what it guards); stage 4 should not claim more. | informational |

## Verdict
**stage 3 ACCEPTED.** No gold or fold-k study reached any training set; OOF/bundle contracts hold; gold-58 recomputes exactly
(0.8466); OOF-weak recomputes to 0.8452 vs 0.8463 reported (mcl boundary, F2); acceptance thresholds met (fold-0 weak 0.8495 >= 0.75,
smoke 0.67 > 0.55 in 6.7 min, fold < 1 h, bundle reloads bit-exact, one-missing-slot forward finite). F1 should be addressed
(or at least the last-epoch OOF reported) before main-arm decisions in stage 6.