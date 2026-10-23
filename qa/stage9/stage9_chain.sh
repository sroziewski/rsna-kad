#!/usr/bin/env bash
# Stage 9 honest pseudo-label experiment, queued behind the armA8 chain on GPU 0.
#   srcA (folds 1,2) -> labels folds 3,4 ; srcB (folds 3,4) -> labels folds 1,2 ;
#   neither sees fold 0 -> data/targets/v2/fold0.parquet -> arm A e8 on fold 0 -> compare.
set -euo pipefail
P=/raid/sroziewski/sandbox/bin/python
cd /raid/sroziewski/sandbox/rsna_kad

echo "waiting for the armA8 chain to release GPU 0 ..."
while screen -ls | grep -q '[.]armA8[[:space:]]'; do sleep 300; done
# and for the GPU to actually drain (the process can outlive the screen by seconds)
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$used" -lt 8000 ] && break
  sleep 60
done
echo "GPU 0 free (${used} MiB used) at $(date -Is); starting stage 9"

$P scripts/train.py --config configs/stage9_pseudo_srcA.yaml --fold 0
$P scripts/train.py --config configs/stage9_pseudo_srcB.yaml --fold 0
$P scripts/make_pseudo_targets.py build
$P scripts/train.py --config configs/main_armA8_v2.yaml --fold 0
$P scripts/compare_runs.py --runs runs/main_convnext_s_384_e8 runs/main_armA8_v2 \
   --fold 0 --targets v1 --out qa/stage9/compare_v2.md
echo "stage 9 done at $(date -Is)"
