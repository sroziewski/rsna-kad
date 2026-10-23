#!/usr/bin/env bash
# Screen 'a_done' (coordinator 2026-09-30): pipeline A finishes before pipeline B gets a GPU.
# Waits for the exact screens of every A-side job - pw_gpu0, pw_gpu1 (pairwise), tta2_gpu0 (TTA views),
# mri_gpu0, mri_gpu1 (MRI-aug stage 1) and mri_s2 (MRI judge + optional stage 2) - then runs the pairwise
# judge and writes qa/A_DONE (one line, timestamp) only if the pairwise judge, the TTA scoring and the
# MRI-aug stage (qa/mriaug/STAGE_DONE) all completed. Pipeline B's chains wait for that file.
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
for s in pw_gpu0 pw_gpu1 tta2_gpu0 mri_gpu0 mri_gpu1 mri_s2; do
  while screen -ls | grep -q "[.]${s}[[:space:]]"; do sleep 120; done
  echo "$(date -Is) screen $s gone"
done
export CUDA_VISIBLE_DEVICES=
ok=1
$P qa/full/compare_pw.py > qa/full/compare_pw.txt 2>&1 || ok=0
cat qa/full/compare_pw.txt
grep -q "ALLDONE tta2" logs/tta2_gpu0_*.log && [ -f qa/tta2/compare.txt ] || { echo "TTA chain did not finish"; ok=0; }
grep -q "^ok" qa/mriaug/STAGE_DONE 2>/dev/null || { echo "MRI-aug stage did not finish"; ok=0; }
if [ $ok = 1 ]; then
  date -Is > qa/A_DONE
  echo "$(date -Is) wrote qa/A_DONE"
else
  echo "$(date -Is) NOT writing qa/A_DONE: a step failed - see above"
fi
