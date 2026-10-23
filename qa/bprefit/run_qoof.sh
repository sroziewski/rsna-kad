#!/usr/bin/env bash
# bprefit step 1 on our server: fake /kaggle tree as qa/andrey/run_mthoof_local.sh, then qa/bprefit/qoof_driver.py with
# bpblur's patched Qile script (blur 0.65 after the warp) and B's unpatched one (parity reference), /kaggle/ remapped.
#   CUDA_VISIBLE_DEVICES=1,3 scripts/run_screen.sh qoof bash qa/bprefit/run_qoof.sh
set -euo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
R=$P/.tmp/bprefit
F=$R/fake
EXT=/raid/sroziewski/data/rsna_kad_ext/qilejiang76
PY=/raid/sroziewski/sandbox/bin/python
SHARP=$P/.tmp/mthoof/ours_mth_extracted.py
BLUR=$P/qa/bblur/kernel_bpblur/run_ours_patched.py
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
grep -q "^K1_BLUR_SIGMA = 0.65 " $BLUR
[ "$(grep -c /kaggle/ $SHARP)" = "$(grep -c /kaggle/ $BLUR)" ] || { echo "/kaggle/ counts differ"; exit 1; }
rm -rf $F
mkdir -p $F/kaggle/working $F/kaggle/input
ln -s /raid/sroziewski/data/rsna_kad $F/kaggle/input/rsna-knee-abnormality-detection
for d in rsna-knee-lineage-g1-mth-mtr-olp-20261013 rsna-knee-single-models-20261007; do
  mkdir -p $F/kaggle/input/$d
  for f in $EXT/$d/*; do ln -s "$f" $F/kaggle/input/$d/; done
done
sed "s|/kaggle/|$F/kaggle/|g" $SHARP > $R/ours_sharp.py
sed "s|/kaggle/|$F/kaggle/|g" $BLUR > $R/ours_blur.py
cd $F/kaggle/working
$PY -u $P/qa/bprefit/qoof_driver.py --work $F/kaggle/working --comp $F/kaggle/input/rsna-knee-abnormality-detection \
  --folds $P/data/folds/folds_v1.csv --wheel $(ls $EXT/rsna-knee-single-models-20261007/timm-1.0.30-py3-none-any.whl) \
  --ours-sharp $R/ours_sharp.py --ours-blur $R/ours_blur.py \
  --parity-ref $P/.tmp/mthoof/fake/kaggle/working/mth_parity_f0_sharp_MTH_foldmean.csv
