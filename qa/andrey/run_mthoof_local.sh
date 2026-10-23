#!/usr/bin/env bash
# Andrey's andreytikhomirov/rsna-knee-mth-blur-oof (2026-10-20) run on our server instead of a Kaggle T4 session: blurred
# (0.8 px hires) MT-hires OOF on the 4,349 non-gold train studies, for refitting B's stacker. His script runs unchanged
# except "/kaggle/" -> a fake tree under .tmp/mthoof/fake (its embedded ours_mth.py is byte-identical to B's
# run_ours.py, reviewed). Dataset dirs hold per-file symlinks: os.walk does not descend into symlinked dirs.
#   CUDA_VISIBLE_DEVICES=1,3 scripts/run_screen.sh mthoof bash qa/andrey/run_mthoof_local.sh
set -euo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
R=$P/.tmp/mthoof
F=$R/fake
EXT=/raid/sroziewski/data/rsna_kad_ext/qilejiang76
SRC=$P/.tmp/andrey/mthoof/rsna-knee-mth-blur-oof.py
[ "$(grep -c "/kaggle/" $SRC)" = 11 ] || { echo "unexpected /kaggle/ count"; exit 1; }
rm -rf $F
mkdir -p $F/kaggle/working $F/kaggle/input/folds
ln -s /raid/sroziewski/data/rsna_kad $F/kaggle/input/rsna-knee-abnormality-detection
for d in rsna-knee-lineage-g1-mth-mtr-olp-20261013 rsna-knee-single-models-20261007; do
  mkdir -p $F/kaggle/input/$d
  for f in $EXT/$d/*; do ln -s "$f" $F/kaggle/input/$d/; done
done
cp $P/data/folds/folds_v1.csv $F/kaggle/input/folds/
sed "s|/kaggle/|$F/kaggle/|g" $SRC > $R/mthoof_run.py
cd $F/kaggle/working
/raid/sroziewski/sandbox/bin/python -u $R/mthoof_run.py
wc -l $F/kaggle/working/mth_oof_blur_all.csv
