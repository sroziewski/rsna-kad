#!/usr/bin/env bash
# HP-diverse C_p seed ensemble (coordinator 2026-09-27).
#   bash qa/hpe/chain.sh <screen to wait for> <variant n>...   (run as screen hpe_gpu0 / hpe_gpu1)
# Waits on ONE exact screen name, then for enough free GPU memory (co-tenants), then trains each
# variant's 5 folds + predict_oof back to back. Nobody waits on hpe_*, so no mutual wait.
set -u
cd "$(dirname "$0")/../.."
wait_for=$1; shift
NEED_MIB=18000   # C_p peaks at 14.3 GB (runs/main_armc_convnextv2_t_384_p/fold*/train.log) + margin
while screen -ls | grep -q "[.]${wait_for}[[:space:]]"; do sleep 120; done
echo "$(date -Is) screen ${wait_for} gone"
for n in "$@"; do
  while :; do
    read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | tr -d ,)
    free=$((total - used))
    [ "$free" -ge "$NEED_MIB" ] && break
    echo "$(date -Is) GPU $CUDA_VISIBLE_DEVICES free ${free} MiB < ${NEED_MIB}; waiting (co-tenant load)"
    sleep 300
  done
  echo "$(date -Is) starting hpe$n on GPU $CUDA_VISIBLE_DEVICES (free ${free} MiB)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  c=configs/main_armc_p_hpe$n.yaml
  for k in 0 1 2 3 4; do python scripts/train.py --config $c --fold $k || exit 1; done
  python scripts/predict_oof.py --config $c || exit 1
  echo "$(date -Is) ALLDONE hpe$n"
done
