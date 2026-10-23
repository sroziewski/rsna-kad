#!/usr/bin/env bash
# Full-data retrains (coordinator 2026-09-28).
#   bash qa/full/chain.sh <screen to wait for> <filler config or -> <cfg:seed>...   (screens full_gpu0 / full_gpu1)
# Waits on ONE exact screen name. If the full-data mode is not yet marked tested (qa/full/READY, written
# only after the unit tests and a smoke run passed), runs the filler 5-fold config first so the GPU never
# idles, then waits for READY. Each full-data job: train.py --fold full --seed S, run_id <cfg run_id>_full_s<S>.
# Before each job, waits for >= 18 GB free on the GPU (co-tenants). Nobody waits on full_*.
set -u
cd "$(dirname "$0")/../.."
wait_for=$1; filler=$2; shift 2
NEED_MIB=18000
gpu_free() {
  while :; do
    read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | tr -d ,)
    [ $((total - used)) -ge $NEED_MIB ] && { echo "$(date -Is) GPU $CUDA_VISIBLE_DEVICES free $((total - used)) MiB"; return; }
    echo "$(date -Is) GPU $CUDA_VISIBLE_DEVICES free $((total - used)) MiB < $NEED_MIB; waiting"; sleep 300
  done
}
while screen -ls | grep -q "[.]${wait_for}[[:space:]]"; do sleep 120; done
echo "$(date -Is) screen ${wait_for} gone"
if [ ! -f qa/full/READY ] && [ "$filler" != "-" ]; then
  echo "$(date -Is) full-data mode not READY: running filler $filler"
  gpu_free
  for k in 0 1 2 3 4; do python scripts/train.py --config "$filler" --fold $k || exit 1; done
  python scripts/predict_oof.py --config "$filler" || exit 1
  echo "$(date -Is) ALLDONE filler $filler"
fi
while [ ! -f qa/full/READY ]; do echo "$(date -Is) waiting for qa/full/READY"; sleep 300; done
for job in "$@"; do
  cfg=${job%%:*}; seed=${job#*:}
  gpu_free
  echo "$(date -Is) starting full-data $cfg seed $seed"
  python scripts/train.py --config "$cfg" --fold full --seed "$seed" || exit 1
  echo "$(date -Is) ALLDONE full $cfg seed $seed"
done
