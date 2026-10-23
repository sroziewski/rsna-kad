#!/usr/bin/env bash
# Stage 11 / A2: the 5-fold decision. Folds 1-4 of the control and of the variant the fold-0
# triage picked (fold 0 of both already exists - it IS the triage run, same config, same seed).
#
#   ARMN_BEST=residual CUDA_VISIBLE_DEVICES=0 \
#     bash scripts/run_screen.sh armn_full bash qa/stage11/full_chain.sh
#
# Both chains are needed: arm C's published 5-fold numbers were produced at 10 epochs on targets
# v1, so the control (8 epochs, v1_1_gemma3) is the only run that isolates the head. Scoring is
# qa/stage11/compare.sh full, which refuses to run while this screen is alive (§4.1 trap).
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
best=${ARMN_BEST:?set ARMN_BEST to the variant the triage picked (residual|perlabel|noprior)}

for cfg in base "$best"; do
  for k in 1 2 3 4; do
    echo "=== $(date -Is) arm N: $cfg fold $k"
    $P scripts/train.py --config "configs/main_armn_${cfg}_384.yaml" --fold "$k" || exit 1
    rm -f "runs/main_armn_${cfg}_384/fold${k}/last.pt"   # /raid at 92 %
  done
  $P scripts/predict_oof.py --config "configs/main_armn_${cfg}_384.yaml" || exit 1
done
echo "FULLDONE $(date -Is)"
