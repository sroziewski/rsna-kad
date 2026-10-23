#!/usr/bin/env bash
# Gold-58 e2e of the blur patch (qa/blur/patch_legs.py) for legs P (Qile's notebook) and A (hx48 v6), in the fake /kaggle
# tree of qa/v2stack/e2e.sh. One joint per case with V off (--w-partner 1.0) and A as extra leg; the per-leg outputs
# P.csv / A.csv are kept. Cases: orig (unpatched sources), s0 (patched, sigma 0: must equal orig), s<sigma> (the probe).
#   GPU=0 SIGMA=0.65 bash qa/blur/e2e_blur.sh      then python qa/blur/score_legs.py
set -uo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/v2stack
F=$S/fake
B=$P/.tmp/blur
SIGMA=${SIGMA:-0.65}
NB=$P/.tmp/qile/v3nb/rsna-knee-ours-mtr1-mth4-olp5.ipynb
HX=$P/qa/andrey/hx48_ensemble_v6.py
cd $P
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=$S/home TMPDIR=$S/tmp CUDA_VISIBLE_DEVICES=${GPU:-0}
export V2_GPU_REMAP=$CUDA_VISIBLE_DEVICES PATH=$S/bin:$PATH MPLCONFIGDIR=$S/tmp/mpl
PY=$S/venv/bin/python
mkdir -p $B
# real dirs with symlinked files (Qile's os.walk does not follow symlinks; Kaggle mounts real dirs)
for o in qilejiang76/rsna-knee-lineage-g1-mth-mtr-olp-20261013 qilejiang76/rsna-knee-single-models-20261007 \
         andreytikhomirov/rsna-knee-native-hx48-weights andreytikhomirov/rsna-knee-landmark-detector; do
  rm -rf $F/input/datasets/$o; mkdir -p $(dirname $F/input/datasets/$o); cp -rs /raid/sroziewski/data/rsna_kad_ext/$o $F/input/datasets/$o
done
for case in ${CASES:-orig s0 s$SIGMA}; do
  J=$B/e2e_$case; rm -rf $J; mkdir -p $J
  case $case in orig*) nb=$NB; hx=$HX ;; *)   # orig, orig2, ...: unpatched (repeats give the run-to-run noise floor)
    sg=${case#s}; /raid/sroziewski/sandbox/bin/python -I qa/blur/patch_legs.py --sigma $sg --qile-nb $NB --hx48 $HX --out $J/src || exit 1
    nb=$J/src/qile_blur.ipynb; hx=$J/src/hx48_blur.py ;; esac
  # test-only (all cases alike; kept in test_only/, never a kernel source): the wrapper's child GPU = visible index g,
  # mapped through the outer CUDA_VISIBLE_DEVICES so GPU=1 runs on physical GPU 1 (on Kaggle both are 0,1)
  pat="\"CUDA_VISIBLE_DEVICES\": str(g)}"
  # the anchor is on the wrapper line and inside the embedded FILES line; only the wrapper line is rewritten
  [ "$(grep -cF "$pat" $hx)" = 2 ] || { echo "GPU-map anchor count changed in $hx"; exit 1; }
  mkdir -p $J/test_only; cp $hx $J/test_only/hx48_run.py; hx=$J/test_only/hx48_run.py
  sed -i "/^FILES = /! s|\"CUDA_VISIBLE_DEVICES\": str(g)}|\"CUDA_VISIBLE_DEVICES\": os.environ[\"CUDA_VISIBLE_DEVICES\"].split(\",\")[g]}|" $hx
  [ "$(grep -cF "$pat" $hx)" = 1 ] || { echo "GPU-map sed failed"; exit 1; }
  $PY qa/joint/build_joint.py --fake $F --inner $S/kernel_test/blend_v2stack.py --partner-src $nb --w-partner 1.0 \
    --extra A:0.2:$hx:HX48_WORK:hx48 --require-all-legs --out $J/kernel > $J/build.log 2>&1 || { cat $J/build.log; exit 1; }
  tail -1 $J/build.log
  rm -rf $F/working $F/tmp; mkdir -p $F/working $F/tmp
  /usr/bin/time -v -o $J/time.txt env BLEND_WORK=$F/working $PY $J/kernel/joint.py > $J/run.log 2>&1
  echo "== $case: exit $?"; grep "^\[joint" $J/run.log | grep "exit\|DROPPED" | cut -c1-200
  cp -r $F/tmp/joint $J/safe
done
echo BLUR_E2E_DONE
