# Pseudo-label round v3 - teacher -> 0.5/0.5 soft targets -> effnetv2_s student (developer O, 2026-09-25)

Status: targets built and validated; 5-fold chain `pseudo` running on GPU 0 (launched 23:17). Comparison: pending.

## Construction (`labels/pseudo.py` v3 block, `qa/pseudo_v3/make_targets_v3.py`)
- Teacher = equal-weight mean of the sigmoid OOF of main_v3's members: `main_convnext_s_384`, `eff_effv2s_256`,
  `main_armc_convnextv2_t_384` (4,407 rows each; gold dropped before use).
- Label part = `data/targets/v1_1` (4 voters incl. pilkwang), its fold-local calibration kept (JSONs copied).
- `data/targets/v3_pseudo/fold{k}.parquet`, 4,349 rows, gold excluded. Rows of folds != k (the fold-k student's
  training rows): **target = 0.5 x v1_1 + 0.5 x teacher on all 12 labels, every cell incl. silent ones**; weight =
  the §2.1 weight rule re-run with `absent_under_reported` 0.1 -> **0.5** (= `absent_well_reported`; half of the
  target is now an image reading, so a silent cell is as informative as a well-reported `absent`; fixed a priori,
  same for every key). Disagreement factor, hallucination cap and parse-fail/postop overrides unchanged; all
  well-reported weights bit-identical. Fold k's own rows (its validation rows) are **bit-identical to v1_1**, so the
  in-training metric is the v1_1 yardstick. No masking, no rounding (both reported to not help / hurt).
- Checks in the builder (all pass): 4,349 rows, no gold, own rows bit-identical, mixed value = formula (1e-6),
  no well-reported weight moved, no weight fell, 0 training rows without a teacher. Unit tests: 4 new in
  `tests/test_pseudo.py`; full suite 279 green. v1, v1_1, v1_1_gemma3 regenerate **bit-identical** (all 5 folds +
  calibrations, cmp).
- Student config `configs/eff_pseudo_256.yaml` = `eff_effv2s_256.yaml` except run_id and targets_version.

## Leakage
- The teacher OOF of a fold-j study comes from a model trained on folds != j, i.e. on fold k's weak rows. The fold-k
  student trains on those pseudo-targets, so **fold-k weak/strict/consensus OOF is optimistic** (weak is doubly
  unfit: it also penalises departures from report labels).
- **Gold-58 stays clean**: gold rows are in no teacher's and no student's training set - `train.py:266`
  `assert not gold_ids & set(train_ids)` is active (no `-O`, no PYTHONOPTIMIZE in the process env; log: "3480 train
  studies, 880 val (11 gold)") and the targets exclude gold by construction (builder assert).
- Residual, stated: each teacher model's targets were calibrated on the gold rows of its training folds, which
  include fold k's gold. That reaches the fold-k student only as per-(label, state-cell) constants (EB posterior
  from ~46 rows, prior strength 8), never as a study-specific value - a second-order, study-agnostic channel.
  The public LB is fully clean.

## Target movement (training rows, mean over folds)
| key | mean target v1_1 -> v3 | mean abs move | weight v1_1 -> v3 | silent cells re-weighted | silent target | teacher mean |
|---|---|---|---|---|---|---|
| synovitis | 0.349 -> 0.370 | 0.054 | 0.196 -> 0.533 | 2,960 | 0.304 -> 0.337 | 0.391 |
| pf_oa | 0.397 -> 0.400 | 0.108 | 0.683 -> 0.763 | 788 | 0.173 -> 0.257 | 0.403 |
| baker | 0.197 -> 0.189 | 0.051 | 0.480 -> 0.668 | 1,715 | 0.072 -> 0.085 | 0.181 |
| fracture | 0.137 -> 0.141 | 0.037 | 0.460 -> 0.615 | 1,601 | 0.109 -> 0.126 | 0.144 |
| lat_oa | 0.179 -> 0.176 | 0.064 | 0.547 -> 0.677 | 1,266 | 0.098 -> 0.127 | 0.173 |
| contusion | 0.234 -> 0.228 | 0.055 | 0.570 -> 0.674 | 1,130 | 0.133 -> 0.161 | 0.223 |
| med_oa | 0.271 -> 0.275 | 0.074 | 0.614 -> 0.714 | 978 | 0.077 -> 0.127 | 0.278 |
| effusion / med_men / lat_men / acl / mcl | | 0.089 / 0.075 / 0.061 / 0.053 / 0.037 | unchanged | 0 | | |
Full table: `qa/pseudo_v3/target_stats_by_fold.csv`. Main effect on synovitis is the **weight (x2.7)**; its mean
target moves only +0.03 because the teacher is itself under-confident (0.39 vs 47 % gold prevalence).

## Clean pre-check: are the targets closer to gold? (`target_fidelity.py`, all 58, each gold row gets the target it
would get as a training row: own-fold calibration + teacher OOF)
label v1_1 0.845 · teacher 0.904 · **v3 mix 0.915; v3 - v1_1 = +0.070 [+0.045, +0.097], P = 1.00**.
Synovitis 0.635 -> 0.799, effusion 0.774 -> 0.940, fracture 0.803 -> 0.926, pf_oa 0.872 -> 0.931, lat_oa 0.812 -> 0.807.
This is about the targets, not the student; it says the supervision is better, not that the model will be.

## Comparison (pending) - `bash qa/pseudo_v3/compare.sh` after screen `pseudo` has ended (refuses otherwise)
Candidates `eff_effv2s_256` (v1) vs `eff_pseudo_256` (v3_pseudo); arbiters gold58 (primary, clean), consensus,
strict (both llm_v1 labels: same yardstick for both runs, but optimistic for the student), weak on v1 and v1_1
(contaminated). Paired bootstrap 2000.

## Lane extension (coordinator 2026-09-25 23:48)
- Judging changed: consensus is out of every promote decision. E_p (`eff_pseudo_256`) is **kill-only**: kill the lane
  if its gold-58 is below eff_v1 by more than the paired noise (~0.009). Decision = **main_v4 (A_p+E_p+C_p) vs
  main_v3**, paired gold-58 + one paired LB probe: accept at >= +0.004 with gold >= -0.009.
- Queued (waiters exact-match one screen name; nobody waits on `pv3_*`):
  `pv3_Ap` (GPU 0, after `pseudo`): `configs/main_armA8_pseudo.yaml` = `main_convnext_s_384_e8.yaml` + run_id
  `main_convnext_s_384_e8_p` + v3_pseudo. `pv3_Cp` (GPU 1, after `armD`): `configs/main_armc_pseudo.yaml` =
  `main_armc_convnextv2_t_384.yaml` + run_id `main_armc_convnextv2_t_384_p` + v3_pseudo + **epochs 10 -> 8**
  (C_p vs C mixes the schedule change in; armA8 showed 12 -> 8 is a tie). Resolved configs checked against the
  trained runs' config.yaml: no other difference.

## E_p kill check (2026-09-26 22:30, `compare.sh`, paired bootstrap 2000; eff = eff_effv2s_256 on v1)
| arbiter | eff | E_p | paired delta E_p - eff [CI95] |
|---|---|---|---|
| **gold58 (clean)** | 0.8666 | 0.8999 | **+0.0328 [+0.0126, +0.0538], P = 1.00** |
| strict (optimistic) | 0.8388 | 0.8767 | +0.0379 [+0.0347, +0.0413] |
| consensus (optimistic, not a decision input) | 0.8615 | 0.8994 | +0.0379 [+0.0337, +0.0425] |
| weak v1 / v1_1 (CONTAMINATED) | 0.9080 / 0.9088 | 0.9425 / 0.9413 | +0.0345 / +0.0325 |
Gold per label, eff -> E_p: synovitis 0.734->0.756, contusion 0.835->0.889, fracture 0.887->0.923, lat_oa 0.743->0.790,
mcl 0.875->0.884, lat_men 0.760->0.845, pf_oa 0.819->0.856, baker 0.947->0.980; all 12 labels up.
**Kill check passes** (the bar was a loss worse than -0.009). Caveat: the teacher beats eff by ~+0.038 gold, so this
gain is compatible with distillation alone; the pseudo-label question is decided by main_v4 vs main_v3.

## E_p export (`export/eff_p_v1`, not pushed)
`export_kaggle.py --run runs/eff_pseudo_256 --mode efficiency --k-infer 16 --no-tta --workers 3 --wall-hours 8.25
--name eff_p_v1 --user sroziews`. Versus eff_v1: ensemble fields equal except members/prior/version (+ new git_sha);
K_infer 16, tta false x5; member json differs only by keys added since (head, label_attn_*, slot_prior_strength,
view_crop_mm), all known to the shipped notebook. Dataset slug set to `sroziews/rsna-kad-weights-effp`.
**Notebook md5 changed on export** (e61dc63c -> 6201e15f): `main` now holds 9a4946d + 75d8c5e (inference code) and
export rebuilds the notebook. I restored it with `git checkout`; a second rebuild by another process at 22:32:52 put
6201e15f back, and I left that one alone. The probe does not depend on it: `qa/pseudo_v3/kernel_effp/` holds its own
kernel-metadata (`sroziews/rsna-kad-effp`, sources `rsna-kad-weights-effp` only) and `kaggle_inference.py` = the
shipped blob from f789876 (md5 e61dc63c). Smoke run of the e61dc63c notebook on the bundle: 3/3 public test studies,
0 prior rows. kernel-metadata.json of the main kernel still f149b39f.

## Round 2 (launched 2026-09-26 22:27)
Teacher = mean sigmoid OOF of E_p, A_p, C_p; `data/targets/v4_pseudo` via the same builder and asserts
(`--version v4_pseudo --teachers ...`; builder re-verified: v3 regenerates bit-identical). Leakage compounds one step
for weak/strict/consensus: the round-1 students trained on targets carrying fold-k weak information, and their OOF
now feeds fold k again. Gold-58 stays clean: gold is in no training set of any stage (teachers, round-1 students,
round-2 students). Screens `pv4_gpu0` (eff_pseudo2_256 then main_convnext_s_384_e8_p2) and `pv4_gpu1`
(main_armc_convnextv2_t_384_p2).
