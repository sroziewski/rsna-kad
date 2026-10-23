#!/usr/bin/env bash
# End-to-end tests of the v2stack kernel on gold-58 in a fake /kaggle tree:
#   1) full blend, 2) heliosli compact bundle missing -> leg 2 fails -> leg 1's file is the submission.
# Runs in a throwaway venv on top of the sandbox one, because the public notebook pip-installs its pinned wheels.
set -uo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/v2stack
F=$S/fake
D=/raid/sroziewski/data/rsna_kad
E=/raid/sroziewski/data/rsna_kad_ext
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=$S/home TMPDIR=$S/tmp CUDA_VISIBLE_DEVICES=${GPU:-1}
mkdir -p $HOME $TMPDIR
if [ ! -x $S/venv/bin/python ]; then
  # the sandbox python is itself a venv, so --system-site-packages would skip its packages: link them with a .pth
  # (appended after the venv's own site-packages, so the notebook's pinned wheels still win)
  /raid/sroziewski/sandbox/bin/python -m venv $S/venv
  $S/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" > $S/purelib
  /raid/sroziewski/sandbox/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" > "$(cat $S/purelib)/00-sandbox.pth"
  # Python 3.12 dropped distutils; the Kaggle image gets it from setuptools' distutils-precedence.pth, and a decoder
  # of the notebook imports it.  A .pth reached through 00-sandbox.pth is not processed, so copy it in (.pth files
  # load alphabetically: 00-sandbox.pth first puts its _distutils_hack on the path).
  cp /raid/sroziewski/sandbox/lib/python3.12/site-packages/distutils-precedence.pth "$(cat $S/purelib)/"
  # The heliosli controller sets CUDA_VISIBLE_DEVICES=<worker index> itself; this wrapper (sys.executable keeps its
  # path via exec -a) sends worker 0 to our pinned GPU instead of the shared GPU 0.
  mv $S/venv/bin/python $S/venv/bin/python-real
  cat > $S/venv/bin/python <<'WRAP'
#!/usr/bin/env bash
[ -n "${V2_GPU_REMAP:-}" ] && [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] && export CUDA_VISIBLE_DEVICES=$V2_GPU_REMAP
exec -a "$0" "$(dirname "$0")/python-real" "$@"
WRAP
  chmod +x $S/venv/bin/python
fi
PY=$S/venv/bin/python
# The heliosli controller starts one worker per `nvidia-smi` GPU, which would spread it over all four shared server
# GPUs.  Report one; the python wrapper above maps that worker to $CUDA_VISIBLE_DEVICES.
export V2_GPU_REMAP=$CUDA_VISIBLE_DEVICES
mkdir -p $S/bin
cat > $S/bin/nvidia-smi <<'SHIM'
#!/usr/bin/env bash
if [ "$*" = "--query-gpu=name --format=csv,noheader" ]; then printf 'NVIDIA RTX A6000
'; exit 0; fi
exec /usr/bin/nvidia-smi "$@"
SHIM
chmod +x $S/bin/nvidia-smi
export PATH=$S/bin:$PATH
cd $P
rm -rf $F; C=$F/input/competitions/rsna-knee-abnormality-detection
mkdir -p $C/test_series $F/working $F/tmp $F/input/datasets/nartaa $F/input/datasets/heliosli
ln -s $E/nartaa_swa $F/input/datasets/nartaa/rsna-knee-publication-swa-weights-20261007
for d in rsna-knee-compact-v1 rsna-knee-compact-r7cnxfsss rsna-orthofoundation-fold0-epoch7-20261004; do
  ln -s $E/heliosli/$d $F/input/datasets/heliosli/$d
done
# EXTRA_DS="owner/slug=dir-under-$E ..." mounts more datasets (qa/r6base)
for x in ${EXTRA_DS:-}; do
  mkdir -p $F/input/datasets/$(dirname ${x%%=*}); ln -s $E/${x#*=} $F/input/datasets/${x%%=*}
done
/raid/sroziewski/sandbox/bin/python -I - "$C" "$D" <<'PYEOF'
import sys, os
from pathlib import Path
import pandas as pd
sys.path[:0] = ["src", "qa/full"]
import compare_pw as c
C, D = sys.argv[1], sys.argv[2]
g = c.gold_frame(Path(D) / "train.csv")
ids = sorted(g["StudyInstanceUID"].astype(str))
assert len(ids) == 58, len(ids)
ts = pd.read_csv(f"{D}/train_series.csv", dtype=str)
ts[ts.StudyInstanceUID.isin(ids)].to_csv(f"{C}/test_series.csv", index=False)
pd.DataFrame({"StudyInstanceUID": ids}).to_csv(f"{C}/test.csv", index=False)
cols = pd.read_csv(f"{D}/sample_submission.csv", nrows=1).columns
sub = pd.DataFrame({"StudyInstanceUID": ids}); [sub.__setitem__(k, 0.5) for k in cols[1:]]
sub.to_csv(f"{C}/sample_submission.csv", index=False)
for s in ids:
    os.symlink(f"{D}/train_series/{s}", f"{C}/test_series/{s}")
print("gold studies", len(ids), "series", ts.StudyInstanceUID.isin(ids).sum())
PYEOF
$PY qa/v2stack/build_kernel.py --fake $F --out $S/kernel_test ${BUILD_ARGS:-} || exit 1
K=$S/kernel_test/blend_v2stack.py
run() {
  rm -rf $F/working $F/tmp; mkdir -p $F/working $F/tmp
  BLEND_WORK=$F/working BLEND_OURS_ARGS="--data-dir $C --weights $P/${OURS_W:-export/main_v5_C}" $PY $K > $S/$1.log 2>&1
  echo "exit $?"; grep "^\[blend" $S/$1.log; rm -rf $S/$1_work; cp -r $F/working $S/$1_work
  [ -d $F/tmp/probe/prediction ] && cp -r $F/tmp/probe/prediction $S/$1_work/_prediction
}
echo "== test 1: full"; run ${TAG:-}e2e_full
[ "${ONLY:-}" = full ] && { echo E2E_DONE; exit 0; }
echo "== test 2: compact bundle missing"
mv $F/input/datasets/heliosli $S/heliosli_hidden
run e2e_nocompact
mv $S/heliosli_hidden $F/input/datasets/heliosli
cmp $S/e2e_nocompact_work/submission.csv $S/e2e_nocompact_work/_ours.csv && echo "FALLBACK_OK: submission == leg 1"
echo E2E_DONE
