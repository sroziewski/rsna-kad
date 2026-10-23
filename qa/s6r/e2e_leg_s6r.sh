#!/usr/bin/env bash
# gold-58 run of qa/s6r/leg_s6r.py (our OURS_s42_fall.pt) in a fake /kaggle tree; = e2e_leg_s6.sh with our dataset layout.
set -uo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/s6r
F=$S/fake_r
D=/raid/sroziewski/data/rsna_kad
E=/raid/sroziewski/data/rsna_kad_ext
PY=${PY:-$P/.tmp/v2stack/venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1 HOME=$S/home TMPDIR=$S/tmp_leg_r CUDA_VISIBLE_DEVICES=${GPU:-1}
mkdir -p $HOME $TMPDIR
cd $P
rm -rf $F; C=$F/input/competitions/rsna-knee-abnormality-detection
W=$F/input/datasets/sroziews/rsna-kad-s6r; M=$S/runs/full
mkdir -p $C/test_series $F/working $F/tmp $W $F/input/datasets/heliosli
# real directories holding symlinked files: some code walks the mounts with os.walk (no followlinks)
cp -rs $E/heliosli/rsna-orthofoundation-fold0-epoch7-20261004 $F/input/datasets/heliosli/
ln -s $E/nartaa_s6fall/checkpoint-manifest.json $W/checkpoint-manifest.json; ln -s $M/run_config.json $W/OURS_s42_fall.run_config.json; ln -s $M/DONE_OURS_s42_fall.json $W/OURS_s42_fall.DONE.json
for f in code_manifest.json s6_extras.bin; do ln -s $S/dl/$f $W/$f; done
ln -s $M/models/OURS_s42_fall.pt $W/OURS_s42_fall.pt
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
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv > $S/leg_s6r_gpu_before.txt
( while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i ${GPU:-1} | sed "s/^/$(date +%T) /"; sleep 30; done ) > $S/leg_s6r_gpu_trace.txt 2>&1 &
MON=$!
t0=$(date +%s)
K1_KAGGLE_ROOT=$F K1_TMP=$F/tmp $PY $P/qa/s6r/leg_s6r.py > $S/leg_s6r_gold.log 2>&1
rc=$?
echo "LEG_EXIT $rc seconds $(( $(date +%s) - t0 ))" | tee -a $S/leg_s6r_gold.log
kill $MON
[ -f $F/working/submission.csv ] && cp $F/working/submission.csv $S/gold_s6r.csv
echo E2E_DONE
