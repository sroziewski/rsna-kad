#!/usr/bin/env bash
# Gold-58 e2e of the R7 blur (qa/blur/patch_r7.py) through the V leg: qa/v2stack/e2e.sh (ONLY=full) with the patched
# v3stack notebook, for sigma 0 mm (parity with the unpatched run .tmp/v2stack/v3_e2e_full_work) and SIGMA_MM.
# e2e.sh rebuilds .tmp/v2stack/kernel_test, the inner kernel the other gold harnesses use: it is saved first and
# restored at the end.          GPU=1 SIGMA_MM=0.288 bash qa/blur/run_r7.sh
set -uo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/v2stack
B=$P/.tmp/blur/r7
NB=$P/qa/v3stack/src/0-957-latest-dataset-score-in-one-click.ipynb
cd $P
mkdir -p $B
# an earlier aborted run left its saved copy: that is the original, never overwrite it with a patched build
[ -d $B/kernel_test_saved ] || cp -a $S/kernel_test $B/kernel_test_saved || exit 1
want=$(md5sum < $B/kernel_test_saved/blend_v2stack.py)
trap 'rm -rf $S/kernel_test; cp -a $B/kernel_test_saved $S/kernel_test; [ "$(md5sum < $S/kernel_test/blend_v2stack.py)" = "$want" ] && rm -rf $B/kernel_test_saved && echo "kernel_test restored" || echo "kernel_test RESTORE FAILED (copy kept in $B/kernel_test_saved)"' EXIT
for sg in ${CASES:-0 ${SIGMA_MM:-0.288}}; do
  out=$B/nb_s$sg.ipynb
  /raid/sroziewski/sandbox/bin/python -I qa/blur/patch_r7.py --sigma-mm $sg --nb $NB --out $out | tee $B/patch_s$sg.txt || exit 1
  sha=$(sha256sum $out | cut -c1-64)
  rm -rf $S/v3blur${sg}_e2e_full_work   # never score a stale run if this build fails
  GPU=${GPU:-1} ONLY=full TAG=v3blur${sg}_ BUILD_ARGS="--nb $out --nb-sha $sha" bash qa/v2stack/e2e.sh > $B/e2e_s$sg.log 2>&1
  echo "== sigma $sg mm: $(grep -m1 "^exit " $B/e2e_s$sg.log)"; grep "^\[blend" $B/e2e_s$sg.log | tail -3
done
echo R7_BLUR_DONE
