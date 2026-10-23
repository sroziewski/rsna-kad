#!/usr/bin/env bash
# Stage 11 / A2 triage on GPU 0, back to back so the GPU never idles (the 200-study smoke of all
# three heads ran on 2026-09-23: 299 img/s, 14.0 GB each; train loss 0.7161 / 0.7095 / 0.7281):
#   fold 0 of control / per_label_residual / per_label_attn / best-without-prior.
# Fold-0 deltas are INDICATIVE ONLY (§4.3: arm C read +0.0075 on fold 0, +0.0039 assembled).
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_screen.sh armn_triage bash qa/stage11/triage.sh
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"


for cfg in base residual perlabel noprior; do
  echo "=== $(date -Is) arm N: $cfg fold 0"
  $P scripts/train.py --config "configs/main_armn_${cfg}_384.yaml" --fold 0 || exit 1
  # /raid is at 91 %: the resume checkpoint is useless once the fold is done.
  rm -f "runs/main_armn_${cfg}_384/fold0/last.pt"
done
echo "TRIAGEDONE $(date -Is)"
