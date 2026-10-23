#!/usr/bin/env bash
# Stage 11 / A2 triage on targets v3_pseudo (2026-09-26), GPU 0, back to back. The control is C_p
# (runs/main_armc_convnextv2_t_384_p, fold 0 already trained with the identical recipe), so only
# the three head variants run here. Queued by qa/stage11/queue_v3p.sh behind screen pv4_gpu0.
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
for cfg in residual perlabel noprior; do
  echo "=== $(date -Is) arm N v3p: $cfg fold 0"
  $P scripts/train.py --config "configs/main_armn_${cfg}_v3p_384.yaml" --fold 0 || exit 1
  rm -f "runs/main_armn_${cfg}_v3p_384/fold0/last.pt"   # /raid at ~91 %
done
echo "TRIAGEDONE $(date -Is)"
