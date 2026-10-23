#!/usr/bin/env bash
# Gold-58 e2e of the three-leg joint kernel (qa/andrey, 2026-10-15): V = v3stack, P = Qile's notebook v2 (fast globs),
# A = Andrey's hx48_ensemble.py (LB 0.948) at w 0.2, run unchanged with HX48_WORK=<work>/hx48. Reuses the fake /kaggle tree
# and the fake-path v3stack kernel of qa/joint/e2e_joint.sh. Cases: ok (all legs) and afail (A replaced by a script
# that exits 1: the submission must equal the V/P blend). Then qa/andrey/check_joint3.py scores and checks.
set -uo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/v2stack
F=$S/fake
J=$P/.tmp/andrey/e2e
cd $P
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=$S/home TMPDIR=$S/tmp CUDA_VISIBLE_DEVICES=${GPU:-0}
export V2_GPU_REMAP=$CUDA_VISIBLE_DEVICES PATH=$S/bin:$PATH MPLCONFIGDIR=$S/tmp/mpl
PY=$S/venv/bin/python
C=$F/input/competitions/rsna-knee-abnormality-detection
rm -rf $J; mkdir -p $J
printf 'raise SystemExit(1)\n' > $J/fail_leg.py
for case in ok afail; do
  src=qa/andrey/hx48_ensemble.py; [ $case = afail ] && src=$J/fail_leg.py
  $PY qa/joint/build_joint.py --fake $F --inner $S/kernel_test/blend_v2stack.py --partner-fast-glob \
    --partner-src qa/qile/qile_v2.ipynb --extra A:0.2:$src:HX48_WORK:hx48 --out $J/kernel_$case | tail -2
  rm -rf $F/working $F/tmp; mkdir -p $F/working $F/tmp
  BLEND_WORK=$F/working BLEND_OURS_ARGS="--data-dir $C --weights $P/export/main_v5_C" $PY $J/kernel_$case/joint.py > $J/$case.log 2>&1
  echo "== $case: exit $?"; grep "^\[joint" $J/$case.log
  cp -r $F/working $J/${case}_work; cp -r $F/tmp/joint $J/${case}_safe
done
CUDA_VISIBLE_DEVICES= /raid/sroziewski/sandbox/bin/python qa/andrey/check_joint3.py $J
echo JOINT3_E2E_DONE
