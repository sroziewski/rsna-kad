#!/usr/bin/env bash
# Task 1 verdict: the assembled 8-epoch OOF against arm A's 12-epoch OOF, on identical rows and
# the identical targets v1 tables, plus each run's per-fold epoch curve. Waits for the armA8
# chain (5 folds + predict_oof) to release, then writes qa/stage9/compare_armA_e8.md.
set -euo pipefail
P=/raid/sroziewski/sandbox/bin/python
cd /raid/sroziewski/sandbox/rsna_kad
while screen -ls | grep -q '[.]armA8[[:space:]]'; do sleep 300; done
OUT=qa/stage9/compare_armA_e8.md
$P scripts/compare_runs.py --runs runs/main_convnext_s_384 runs/main_convnext_s_384_e8 \
   --targets v1 --out "$OUT"
{
  echo
  echo '## Per-fold epoch curves'
  echo
  echo '### arm A (12 epochs)'
  echo '```'
  $P qa/stage9/fold_curves.py runs/main_convnext_s_384
  echo '```'
  echo
  echo '### arm A e8 (8 epochs)'
  echo '```'
  $P qa/stage9/fold_curves.py runs/main_convnext_s_384_e8
  echo '```'
} >> "$OUT"
echo "wrote $OUT"
