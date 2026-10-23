#!/usr/bin/env bash
# Pseudo round 2, GPU 0: E_p2 (effnet, first read) then A_p2, back to back. Screen 'pv4_gpu0'.
set -eu
cd "$(dirname "$0")/../.."
for c in eff_pseudo2_256 main_armA8_pseudo2; do
  for k in 0 1 2 3 4; do python scripts/train.py --config configs/$c.yaml --fold $k; done
  python scripts/predict_oof.py --config configs/$c.yaml
  echo "$(date -Is) ALLDONE $c"
done
