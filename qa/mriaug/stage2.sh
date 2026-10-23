#!/usr/bin/env bash
# Screen 'mri_s2' (coordinator 2026-09-30): after the exact screens mri_gpu0 and mri_gpu1 (MRI-aug stage 1,
# seeds 2026/2126 and 2226) end, run the pre-declared judge; if the 3-seed paired gold-58 delta is
# >= +0.002, train stage 2 (seeds 2326, 2426 on GPU 0 as screen mri2_gpu0; 2526 on GPU 1 as mri2_gpu1)
# and wait for both. Writes qa/mriaug/STAGE_DONE ("ok <verdict>") only when everything it started has
# finished cleanly; a_done waits for this screen and checks that file. Nobody else waits on mri_*.
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
CFG=configs/main_armc_pseudo_full_mri.yaml
for s in mri_gpu0 mri_gpu1; do
  while screen -ls | grep -q "[.]${s}[[:space:]]"; do sleep 120; done
  echo "$(date -Is) screen $s gone"
done
rm -f qa/mriaug/STAGE_DONE
CUDA_VISIBLE_DEVICES= $P qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_mri_s{} --bar 0.002 --p-bar 0 \
  --out qa/mriaug/compare_mri.json > qa/mriaug/compare_mri.txt 2>&1 || { cat qa/mriaug/compare_mri.txt; echo "judge failed"; exit 1; }
cat qa/mriaug/compare_mri.txt
delta=$($P -c "import json;print(json.load(open('qa/mriaug/compare_mri.json'))['pairs']['new_mean3 - base_mean3']['delta'])")
if $P -c "import sys; sys.exit(0 if $delta >= 0.002 else 1)"; then
  echo "$(date -Is) stage-1 delta $delta >= +0.002: starting stage 2"
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_screen.sh mri2_gpu0 bash qa/full/after_cfg.sh - 0 $CFG:2326 $CFG:2426
  CUDA_VISIBLE_DEVICES=1 bash scripts/run_screen.sh mri2_gpu1 bash qa/full/after_cfg.sh - 1 $CFG:2526
  sleep 30
  for s in mri2_gpu0 mri2_gpu1; do
    while screen -ls | grep -q "[.]${s}[[:space:]]"; do sleep 120; done
  done
  bad=$(CUDA_VISIBLE_DEVICES= $P -c "
import sys; sys.path.insert(0, 'src'); from rsna_kad import runguard
print(' '.join(p for s in (2326, 2426, 2526) for p in runguard.fold_problems(f'runs/main_armc_convnextv2_t_384_p_full_mri_s{s}/full')))")
  [ -z "$bad" ] || { echo "stage 2 incomplete: $bad"; exit 1; }
  echo "ok stage2-trained delta=$delta" > qa/mriaug/STAGE_DONE
else
  echo "ok tie delta=$delta (stage 2 not triggered)" > qa/mriaug/STAGE_DONE
fi
echo "$(date -Is) $(cat qa/mriaug/STAGE_DONE)"
