# Full-data retrains (developer O, 2026-09-28)

## Mode
- `scripts/train.py --fold full [--seed S]`: trains on every non-gold study with a targets row (4,349), evaluates on
  the 58 gold only (held out: gold is in no training set; `train.py`'s gold assert is unchanged and covers it).
  Same schedule, `select_epoch: last` enforced (with gold as the only held-out rows, "best" would select on gold).
  Bundle `fold: "full"`, run dir `runs/<run_id>/full`; `--seed S` without `--run-id` writes `runs/<run_id>_s<S>/`.
- Targets `data/targets/v3_pseudo_full/full.parquet` (`qa/full/make_targets_full.py`): v1_1 recipe with the EB
  calibration fitted on all 58 gold (the recipe first reproduces v1_1 fold 0 bit-for-bit from its own calibration),
  then 0.5 x label + 0.5 x main_v3 teacher OOF on every cell, silent under-reported weight 0.5; 4,349 rows, no gold.
- Export: `runguard.member_dirs` ships `<run>/full` as the run's single, complete member; every existing refusal
  (live, last.pt, not at final epoch, smoke/limit, missing fold of a fold run) still applies, plus "full/ next to
  fold dirs". `kaggle_export.member_name` = `<run_id>_full`. `infer.py` unchanged (a test runs a full member
  through `members_from_ensemble` + `predict_studies`). Notebook not rebuilt.
- Liveness (runguard): a run is live only if a live `train.py`/`predict_oof.py` writes its run id (via --run-id,
  or its --config's run_id, +`_s<seed>`), or a process holds a file in its dir open. Merely naming it (chain
  script, comment, waiter, scorer) no longer counts; test reproduces the 2026-09-27 C_p/hpe-chain refusal.
- Tests: `tests/test_full_data.py` (new), `tests/test_runguard.py` (two liveness tests rewritten to the new
  rule, three added). Full suite 347 tests, 0 failures (the one skip is the stale-notebook xfail).
- GPU smoke: `eff_pseudo_full --fold full --seed 7 --limit 40 --epochs 1`: fold "full", 58 gold scored, 3.6 GB.

## Queue (waiters wait on one exact screen name; qa/full/chain.sh; READY written 17:12, so no filler)
- `full_gpu1` after `hpe_gpu1`: C_p full seeds 2026, 2126, 2226, then A_p full seed 2026.
- `full_gpu0` after `hpe_gpu0`: E_p full seeds 2026, 2126, 2226, then A_p full seed 2126.

## Pairwise ranking loss (queued 2026-09-30 22:51)
- `TrainConfig.loss` = `bce` (default, path unchanged) | `bce_pairwise`; `pairwise_weight` 0.12 (competitor's value,
  not tuned). Loss = (1 - w) x masked soft BCE + w x mean over labels of softplus(-(z_pos - z_neg)); aux heads plain BCE.
  Pos/neg = metrics.binarise (target >= 0.6; target <= 0.2 with weight >= 0.5). `train.PairwiseRankLoss`.
- Measured on v3_pseudo_full, 100 random 4-study batches, in-batch pairs only: share of batches with >= 1 valid pair
  acl .34, mcl .06, med_men .75, lat_men .35, med_oa .46, lat_oa .21, pf_oa .48, effusion .54, synovitis .01,
  baker .44, contusion .15, fracture .08 (3.9 of 12 labels per batch). Too sparse for the rare labels, so a
  per-label bank of the last 64 detached positive and 64 negative logits is used (bank-bank pairs excluded).
  With the bank (last 100 batches of epoch 1): >= 1 pair in 98-100 % of batches for 11 labels, synovitis 31 %
  (only 40 synovitis negatives exist in the table); mcl has only 20 positives, so its positive bank is old logits.
  The bank starts empty after a resume.
- Tests `tests/test_pairwise_loss.py` (7): defaults; pair rule == binarise; large margin -> ~0 and no pair -> exact 0
  with zero gradient; finite gradient in the right direction; bank supplies pairs; unknown loss rejected;
  end-to-end CPU training with bce_pairwise at weight 0 gives bit-identical weights to the default bce run
  (and weight 0.12 differs). Suite 354 tests, 0 failures.
- `configs/main_armc_pseudo_full_pw.yaml` = main_armc_pseudo_full.yaml + run_id + `loss: bce_pairwise`.
  `pw_gpu0` (after cb_gpu0): seed 2226; `pw_gpu1` (after cb_gpu1): seeds 2026, 2126.
- Judge `qa/full/compare_pw.py` (pre-declared): 3-seed mean pw vs 3-seed mean Cfull (same seeds), gold-58 paired
  bootstrap 2000, accept only at >= +0.004 with P >= 0.95, else tie; per-seed pairs and per-label deltas.
