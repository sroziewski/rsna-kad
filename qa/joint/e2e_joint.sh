#!/usr/bin/env bash
# Gold-58 e2e of the joint kernel (qa/joint, 2026-10-14) with a stand-in partner (qa/joint/standin_partner.py):
# qa/v2stack/e2e.sh builds the fake /kaggle tree and the fake-path v3stack kernel (and runs it once as the reference),
# then the joint kernel runs: ok, ok with the partner first, partner fails, partner misses a study, partner times out.
set -uo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/v2stack
F=$S/fake
J=$P/.tmp/joint
cd $P
V3="--nb qa/v3stack/src/0-957-latest-dataset-score-in-one-click.ipynb --nb-sha cb2a0f57962ca9b6668661e34461aad772ed9ed10d576d32772e7c2898f2e315 --base-datasets heliosli/rsna-knee-compact-r7cnxfsss nartaa/rsna-knee-publication-swa-weights-20261007 heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
GPU=${GPU:-1} ONLY=full TAG=joint_v3_ BUILD_ARGS="$V3" bash qa/v2stack/e2e.sh || exit 1
# the environment qa/v2stack/e2e.sh runs its kernel in
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=$S/home TMPDIR=$S/tmp CUDA_VISIBLE_DEVICES=${GPU:-1}
export V2_GPU_REMAP=$CUDA_VISIBLE_DEVICES PATH=$S/bin:$PATH
PY=$S/venv/bin/python
C=$F/input/competitions/rsna-knee-abnormality-detection
rm -rf $J; mkdir -p $J $F/input/datasets/standin/partner-standin
CUDA_VISIBLE_DEVICES= /raid/sroziewski/sandbox/bin/python - $F/input/datasets/standin/partner-standin/preds.csv <<'PYEOF'
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path[:0] = ["src", "qa/full"]
import compare_pw as c
from rsna_kad.constants import DATA_DIR, ID_COL
g = c.gold_frame(Path(DATA_DIR) / "train.csv")
ids = sorted(g[ID_COL].astype(str))
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])
p = np.mean([c.load(f"main_armc_convnextv2_t_384_p_full_oai_s{s}", ids) for s in range(2026, 2600, 100)], axis=0)
df = pd.DataFrame(p, columns=cols); df.insert(0, ID_COL, ids); df.to_csv(sys.argv[1], index=False)
print("standin preds", df.shape)
PYEOF
case_run() {  # name, STANDIN_MODE, extra build args
  $PY qa/joint/build_joint.py --fake $F --inner $S/kernel_test/blend_v2stack.py \
    --partner-src qa/joint/standin_partner.py --out $J/kernel_$1 ${3:-} | tail -1
  rm -rf $F/working $F/tmp; mkdir -p $F/working $F/tmp
  STANDIN_MODE=$2 BLEND_WORK=$F/working BLEND_OURS_ARGS="--data-dir $C --weights $P/export/main_v5_C" \
    $PY $J/kernel_$1/joint.py > $J/$1.log 2>&1
  echo "== $1: exit $?"; grep "^\[joint" $J/$1.log
  cp -r $F/working $J/$1_work; cp -r $F/tmp/joint $J/$1_safe
}
case_run ok ok
case_run pfirst ok --partner-first
case_run fail fail
case_run partial partial
case_run sleep sleep "--partner-max-h 0.01"
CUDA_VISIBLE_DEVICES= /raid/sroziewski/sandbox/bin/python qa/joint/check_joint.py $J $S/joint_v3_e2e_full_work
echo JOINT_E2E_DONE
