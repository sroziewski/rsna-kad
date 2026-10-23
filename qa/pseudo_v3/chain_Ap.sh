#!/usr/bin/env bash
# Pseudo-label lane Ap (coordinator 2026-09-25): wait for screen 'pseudo' to exit, then 5 folds of
# configs/main_armA8_pseudo.yaml + predict_oof. Launched as screen 'pv3_Ap'; exact-name match on 'pseudo' only,
# and no other waiter matches 'pv3_*', so it cannot enter a mutual wait.
set -u
cd "$(dirname "$0")/../.."
while screen -ls | grep -q "[.]pseudo[[:space:]]"; do sleep 120; done
echo "$(date -Is) screen pseudo gone; starting Ap on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"
set -e
for k in 0 1 2 3 4; do python scripts/train.py --config configs/main_armA8_pseudo.yaml --fold $k; done
python scripts/predict_oof.py --config configs/main_armA8_pseudo.yaml
echo "$(date -Is) ALLDONE Ap"
