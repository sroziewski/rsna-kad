#!/usr/bin/env bash
# joint3blurall v2 (2026-10-20): 57409437 (joint3blurall) ERRORed on the hidden test, most likely the last leg (A)
# timed out / was skipped under --require-all-legs (R7 blur: +0 s on gold, cannot fail V). v2 buys time: A without
# blur (hx48 v6 as in joint3v6; gold -0.0006, ~ -13 min hidden) and a joint budget of 8.8 h instead of 8.6 h; and it
# is a probe, so --diag-encode replaces --require-all-legs (a dropped leg shows in the score, not as an error).
set -euo pipefail
cd /raid/sroziewski/sandbox/rsna_kad
PY=/raid/sroziewski/sandbox/bin/python
T=qa/blur/src_s0.65
$PY qa/joint/build_joint.py --inner qa/blur/v_inner_s0.288/blend_v2stack.py --partner-src $T/qile_blur.ipynb \
  --partner-datasets qilejiang76/rsna-knee-lineage-g1-mth-mtr-olp-20261013 qilejiang76/rsna-knee-single-models-20261007 \
  --extra A:0.2:qa/andrey/hx48_ensemble_v6.py:HX48_WORK:hx48 \
  --extra-datasets andreytikhomirov/rsna-knee-native-hx48-weights andreytikhomirov/rsna-knee-landmark-detector \
  --diag-encode --total-h 8.8 --kernel-id sroziews/rsna-kad-joint3blurall2 --out qa/blur/kernel_joint3blurall2
