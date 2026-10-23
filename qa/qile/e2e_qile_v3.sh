#!/usr/bin/env bash
# Gold-58 parity of Qile's faster notebook (pulled 2026-10-17, decode overlapped with GPU; "predictions stay the same"):
# leg P alone (--w-partner 1.0 = V not run) in the fake /kaggle tree of qa/v2stack/e2e.sh, compared with leg P of the
# 2026-10-15 fast-glob e2e (old notebook v2). Wall time and peak RSS of the leg are printed too.
set -uo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/v2stack
F=$S/fake
J=$P/.tmp/qile/e2e_v3
NB=${NB:-$P/.tmp/qile/v3nb/rsna-knee-ours-mtr1-mth4-olp5.ipynb}
cd $P
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=$S/home TMPDIR=$S/tmp CUDA_VISIBLE_DEVICES=${GPU:-1}
export V2_GPU_REMAP=$CUDA_VISIBLE_DEVICES PATH=$S/bin:$PATH
PY=$S/venv/bin/python
rm -rf $J; mkdir -p $J
# his find_input is os.walk without followlinks: real dirs with symlinked files, as Kaggle mounts real dirs
Q=/raid/sroziewski/data/rsna_kad_ext/qilejiang76
for d in rsna-knee-lineage-g1-mth-mtr-olp-20261013 rsna-knee-single-models-20261007; do
  rm -rf $F/input/datasets/qilejiang76/$d; mkdir -p $F/input/datasets/qilejiang76; cp -rs $Q/$d $F/input/datasets/qilejiang76/$d
done
$PY qa/joint/build_joint.py --fake $F --inner $S/kernel_test/blend_v2stack.py --partner-src $NB \
  --partner-datasets qilejiang76/rsna-knee-lineage-g1-mth-mtr-olp-20261013 qilejiang76/rsna-knee-single-models-20261007 \
  --w-partner 1.0 --out $J/kernel | tail -2
rm -rf $F/working $F/tmp; mkdir -p $F/working $F/tmp
/usr/bin/time -v -o $J/time.txt env BLEND_WORK=$F/working $PY $J/kernel/joint.py > $J/run.log 2>&1
echo "== exit $?"; grep "^\[joint" $J/run.log | tail -4
grep -E "Elapsed|Maximum resident" $J/time.txt
cp -r $F/tmp/joint $J/safe
/raid/sroziewski/sandbox/bin/python -I - $J/safe/P.csv $P/.tmp/qile/e2e_fast/ok_safe/P.csv <<'PYEOF'
import sys
import numpy as np, pandas as pd
a, b = (pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").sort_index() for p in sys.argv[1:3])
assert list(a.index) == list(b.index) and list(a.columns) == list(b.columns), "ids/columns differ"
d = (a - b).abs().to_numpy()
rho = np.mean([a[c].corr(b[c], method="spearman") for c in a.columns])
print(f"P new vs old: n {len(a)}, max abs diff {d.max():.2e}, mean {d.mean():.2e}, spearman {rho:.6f}")
print("PARITY", "PASS" if rho > 0.999 else "FAIL")
PYEOF
echo QILE_V3_E2E_DONE
