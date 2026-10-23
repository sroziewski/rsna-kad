#!/usr/bin/env bash
# qa/ext/worker.sh <gpu> (2026-10-16): one worker per GPU claims the 3 external-knee seeds (atomic mkdir) when the GPU
# has >= 18 GB free. Same seeds as the base and Opus C runs, so the judge compares like with like.
set -u
cd /raid/sroziewski/sandbox/rsna_kad
gpu=$1
P=/raid/sroziewski/sandbox/bin/python
C=qa/ext/claims
mkdir -p $C
# nvidia-smi numbers GPUs in PCI order; make CUDA agree so the memory check looks at the card that trains
export CUDA_DEVICE_ORDER=PCI_BUS_ID
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 0; }
wait_gpu() { while [ "$(free_mb)" -lt 18000 ]; do sleep 120; done; }
for s in 2026 2126 2226; do
  [ -d $C/s_$s ] && continue
  wait_gpu
  mkdir $C/s_$s 2>/dev/null || continue
  echo "$(date -Is) GPU $gpu takes seed $s"
  RSNA_KAD_CACHE=/raid/sroziewski/cache/rsna_kad_ext CUDA_VISIBLE_DEVICES=$gpu $P scripts/train.py \
    --config configs/main_armc_pseudo_full_ext.yaml --fold full --seed $s \
    && echo "ALLDONE ext s$s" || echo "FAILED ext s$s"
done
echo "$(date -Is) GPU $gpu worker done"
