#!/usr/bin/env bash
# bprefit steps 2-4 on the server, after run_qoof.sh: control refit on the original (unblurred) Qile members must
# reproduce B's stacker JSON and give gold d = 0; then members from the blurred qoof outputs -> refit -> gold screen.
#   scripts/run_screen.sh bpfit bash qa/bprefit/run_fit.sh
set -euo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
R=$P/.tmp/bprefit
PY=/raid/sroziewski/sandbox/bin/python
S5=$P/.tmp/andrey/stacker5
WP=/raid/sroziewski/data/rsna_kad_ext/qilejiang76/rsna-knee-lineage-g1-mth-mtr-olp-20261013/OLP_plane_merge_weights.json
B=$P/.tmp/andrey/blurpull/rsna-knee-j3-stack-blur/code/rsna-knee-j3-stack-blur.py
export PYTHONDONTWRITEBYTECODE=1
fit() { $PY -I $S5/fit_stacker5.py --qile-dir "$1" --armc $P/.tmp/andrey/oof_train_armC_5fold.csv --hx $S5/hx16_cv5_oof.csv \
          --labels $S5/hybrid_curated_mrnet_v4_kneemri10_labels.csv --out "$2"; }

echo "== control: refit on the original members vs B's embedded STK"
fit $P/.tmp/qile/in/predictions $R/stk_control.json
$PY -I - "$B" $R/stk_control.json <<'EOF'
import ast, json, sys
import numpy as np
src = open(sys.argv[1]).read()
line = next(l for l in src.splitlines() if l.startswith("STK = json.loads("))
b = json.loads(ast.literal_eval(line[len("STK = json.loads("):line.rindex(")")]))
n = json.load(open(sys.argv[2]))
assert b["features"] == n["features"], "feature order differs"
d = max(np.abs(np.array(b[k]) - np.array(n[k])).max() for k in ("mean", "std"))
d = max([d] + [np.abs(np.array(b[k][f]) - np.array(n[k][f])).max() for k in ("lr", "ranklr") for f in b["lr"]])
print(f"control max |coef diff| vs B: {d:.2e}")
sys.exit(0 if d < 1e-6 else 1)
EOF

echo "== step 1: members from the blurred qoof outputs"
$PY -I $P/qa/bprefit/make_members.py --work $R/fake/kaggle/working --weights $WP \
  --labels /raid/sroziewski/data/rsna_kad/sample_submission.csv --out $R/members

echo "== step 2: refit on the blurred members"
fit $R/members $R/stk_refit.json

echo "== step 3: gold-58 screen (control first: must be d +0.0000)"
$PY -I $P/qa/bprefit/gold_screen.py $R/members $R/stk_control.json
$PY -I $P/qa/bprefit/gold_screen.py $R/members $R/stk_refit.json
echo FIT_DONE
