#!/usr/bin/env bash
# Screen 'opus_judge' (2026-10-15): after both workers end, run the pre-declared judges (docs/architecture.md,
# 2026-10-15 Opus-label entry) if all three seeds finished.
set -u
cd /raid/sroziewski/sandbox/rsna_kad
P=/raid/sroziewski/sandbox/bin/python
sleep 60
for s in opus_gpu1 opus_gpu3; do
  while screen -ls | grep -q "[.]${s}[[:space:]]"; do sleep 120; done
  echo "$(date -Is) screen $s gone"
done
for s in 2026 2126 2226; do
  d=runs/main_armc_convnextv2_t_384_p_full_opus_s$s/full
  { [ -f $d/oof.parquet ] && [ -f $d/model.safetensors ] && [ ! -f $d/last.pt ]; } || { echo "seed $s incomplete - no verdict"; exit 1; }
done
CUDA_VISIBLE_DEVICES= $P qa/full/compare_pw.py --new main_armc_convnextv2_t_384_p_full_opus_s{} \
  --out qa/opus/compare.json > qa/opus/compare.txt 2>&1; echo "solo judge rc=$?"
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 $P qa/opus/gold_joint_opus.py > qa/opus/joint.txt 2>&1; rc=$?
echo "joint judge rc=$rc"; cat qa/opus/compare.txt qa/opus/joint.txt
[ $rc = 0 ] && date -Is > qa/opus/DONE
echo "$(date -Is) ALLDONE opus judge"
