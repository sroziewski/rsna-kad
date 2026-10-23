#!/usr/bin/env bash
# Screen 'tta2_gpu0' (coordinator 2026-09-30): wait for the exact screen pw_gpu0, then >= 18 GB free on
# the GPU, then the four views in the pre-declared order (base, crop first so (a)+(b) land even if cut
# short). Stops unless base reproduces the run's own OOF. Nobody waits on tta2_*.
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
while screen -ls | grep -q "[.]pw_gpu0[[:space:]]"; do sleep 120; done
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -lt 18000 ]; do
  echo "$(date -Is) waiting for 18 GB free"; sleep 120; done
echo "$(date -Is) start on GPU $CUDA_VISIBLE_DEVICES"
set -e
$P qa/tta2/tta2_oof.py --views base
$P qa/tta2/tta2_oof.py --check | tee qa/tta2/base_check.txt
$P - <<'PY'
import re, sys
m = float(re.search(r"max \|d\| ([0-9.e+-]+)", open("qa/tta2/base_check.txt").read()).group(1))
sys.exit(0 if m < 0.01 else f"base does not reproduce the run's OOF (max |d| {m}); stopping")
PY
$P qa/tta2/tta2_oof.py --views crop
$P qa/tta2/tta2_oof.py --views rotp5 rotm5
CUDA_VISIBLE_DEVICES= $P qa/tta2/compare.py | tee qa/tta2/compare.txt
echo "$(date -Is) ALLDONE tta2"
