#!/usr/bin/env bash
# s6r (2026-10-18, user-requested): nartaa's released S6 all-data trainer (nartaa/rsna-knee-s6fall-weights,
# training_code/s6, Apache-2.0) retrained on OUR labels. Builds the S6F0 layout their scripts expect in .tmp/s6r/S6F0:
# their s6 scripts (copies; only the /home/danial paths repointed), heliosli's six sources + MONAI 0.6.0 fixes from
# s6_extras.bin, heliosli's cloud_code from rsna-orthofoundation-fold0-epoch7-20261004, folds_v0.json, the timm
# convnext_small init. Then our targets (qa/s6r/make_targets_s6r.py). Idempotent; never touches the originals.
set -euo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
X=/raid/sroziewski/data/rsna_kad_ext
SRC=$X/nartaa_s6fall/training_code/s6
H=$X/heliosli/rsna-orthofoundation-fold0-epoch7-20261004
R=$P/.tmp/s6r
W=$R/S6F0
PY=/raid/sroziewski/sandbox/bin/python
export HF_HOME=/raid/sroziewski/cache/huggingface
cd $P
rm -rf $W; mkdir -p $W/heliosli $W/data
cp $SRC/*.py $W/
rm $W/make_data.py   # replaced by qa/s6r/make_targets_s6r.py
cp -r $H/cloud_code $W/heliosli/cloud_code
cp $R/ex/heliosli/*.py $W/heliosli/
cp -r $R/ex/cloud_code/. $W/heliosli/cloud_code/
sed -i -e "s#^ANATOMY = .*#ANATOMY = \"$H/anatomy.pt\"#" -e "s#^DATA = .*#DATA = \"/raid/sroziewski/data/rsna_kad\"#" $W/s6common.py
sed -i "s#/home/danial/rsna-knee/research/gap-hunt-20261008/exec/S6F0#$W#" $W/train_s6f0_all.py
! grep -rn "/home/danial" $W/*.py || { echo "unpatched /home/danial path left"; exit 1; }
# heliosli's cloud_code must match the run's pinned hashes (one MONAI file is replaced from s6_extras.bin)
(cd $W/heliosli/cloud_code && sha256sum -c --quiet $R/dl/cloud_code.SHA256SUMS) && echo "cloud_code sha256 OK"
$PY -I -c "import sys; sys.path.insert(0, '$W'); import s6common as SC; SC.verify_copies(anatomy=True); print('sources + anatomy OK')"
# folds_v0.json (nartaa/rsna-knee-hpo-assets) and the ImageNet-22k init the run used
[ -f $R/folds_v0.json ] || { bash scripts/kaggle_cli.sh raw datasets download nartaa/rsna-knee-hpo-assets -f folds_v0.json -p $R >/dev/null;
  [ -f $R/folds_v0.json.zip ] && unzip -o -q $R/folds_v0.json.zip -d $R && rm $R/folds_v0.json.zip; }
sha256sum $R/folds_v0.json
INIT=$($PY -I -c "from huggingface_hub import hf_hub_download as d; print(d('timm/convnext_small.fb_in22k_ft_in1k', 'model.safetensors'))")
echo "INIT $INIT"; sha256sum $INIT
ln -sfn $INIT $R/init_model.safetensors
$PY -I qa/s6r/make_targets_s6r.py
echo SETUP_OK
