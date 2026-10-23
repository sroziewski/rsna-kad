#!/usr/bin/env bash
# qa/opus/worker.sh <gpu> (2026-10-15): one worker per GPU claims the 3 Opus-label seeds (atomic mkdir) when the GPU
# has >= 18 GB free.
set -u
cd /raid/sroziewski/sandbox/rsna_kad
gpu=$1
P=/raid/sroziewski/sandbox/bin/python
C=qa/opus/claims
mkdir -p $C
wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")" -lt 18000 ]; do sleep 120; done; }
for s in 2026 2126 2226; do
  wait_gpu
  mkdir $C/s_$s 2>/dev/null || continue
  echo "$(date -Is) GPU $gpu takes seed $s"
  CUDA_VISIBLE_DEVICES=$gpu $P scripts/train.py --config configs/main_armc_pseudo_full_opus.yaml --fold full --seed $s \
    && echo "ALLDONE opus s$s" || echo "FAILED opus s$s"
done
echo "$(date -Is) GPU $gpu worker done"
