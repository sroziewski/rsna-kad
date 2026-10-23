#!/usr/bin/env bash
# 5-fold of the v3p head the triage picked (folds 1-4; fold 0 is the triage run). Control = C_p.
#   ARMN_BEST=residual CUDA_VISIBLE_DEVICES=0 bash scripts/run_screen.sh armn_full bash qa/stage11/full_chain_v3p.sh
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
best=${ARMN_BEST:?residual|perlabel|noprior}
cfg=configs/main_armn_${best}_v3p_384.yaml
for k in 1 2 3 4; do
  echo "=== $(date -Is) $cfg fold $k"
  $P scripts/train.py --config "$cfg" --fold "$k" || exit 1
  rm -f "runs/main_armn_${best}_v3p_384/fold${k}/last.pt"
done
$P scripts/predict_oof.py --config "$cfg" || exit 1
echo "FULLDONE $(date -Is)"
