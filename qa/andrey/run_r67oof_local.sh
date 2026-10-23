#!/usr/bin/env bash
# Andrey's andreytikhomirov/rsna-knee-r6-r7-blur-oof (2026-10-20) on our server instead of a Kaggle T4 session: blurred
# (0.288 mm DICOM hook) R6 / R7 OOF on the non-gold train studies, hash folds sha1(uid) % 4. One model per process
# and GPU, each in its own fake /kaggle tree (qa/andrey/prep_r67oof.py remaps the paths, incl. the embedded legs).
# Python: the throwaway qa/v2stack venv, because the legs pip-install heliosli's pinned wheels; its wrapper and the
# nvidia-smi shim keep the heliosli controller on our one GPU (qa/v2stack/e2e.sh).
#   scripts/run_screen.sh r7oof bash qa/andrey/run_r67oof_local.sh r7 1
#   scripts/run_screen.sh r6oof bash qa/andrey/run_r67oof_local.sh r6 3
set -euo pipefail
M=${1:?model r6|r7}; GPU=${2:?gpu}
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/r67oof/$M
F=$S/fake
E=/raid/sroziewski/data/rsna_kad_ext
SRC=$P/.tmp/andrey/r67oof/rsna-knee-r6-r7-blur-oof.py
PY=$P/.tmp/v2stack/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=$S/home TMPDIR=$S/tmpdir CUDA_VISIBLE_DEVICES=$GPU V2_GPU_REMAP=$GPU
export PATH=$P/.tmp/v2stack/bin:$PATH
rm -rf $F; mkdir -p $HOME $TMPDIR $F/kaggle/working $F/kaggle/input $F/tmp
ln -s /raid/sroziewski/data/rsna_kad $F/kaggle/input/rsna-knee-abnormality-detection
# real directories holding symlinked files: some code walks the mounts with os.walk (no followlinks)
for d in heliosli/rsna-knee-compact-r7cnxfsss heliosli/rsna-orthofoundation-fold0-epoch7-20261004 rostigor/rsna-knee-r6-fe055406; do
  cp -rs $E/$d $F/kaggle/input/
done
/raid/sroziewski/sandbox/bin/python -I $P/qa/andrey/prep_r67oof.py $SRC $F $M $S/r67oof_run.py
nvidia-smi --query-gpu=name --format=csv,noheader   # the shim: must print one GPU
cd $F/kaggle/working
$PY -u $S/r67oof_run.py
ls -la $F/kaggle/working
