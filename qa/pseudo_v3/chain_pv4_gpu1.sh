#!/usr/bin/env bash
# Pseudo round 2, GPU 1: C_p2. Screen 'pv4_gpu1'.
set -eu
cd "$(dirname "$0")/../.."
for k in 0 1 2 3 4; do python scripts/train.py --config configs/main_armc_pseudo2.yaml --fold $k; done
python scripts/predict_oof.py --config configs/main_armc_pseudo2.yaml
echo "$(date -Is) ALLDONE main_armc_pseudo2"
