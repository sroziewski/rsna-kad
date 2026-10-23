#!/usr/bin/env bash
# s6r chain: wait for the anchor precompute (screen s6r_anchors), freeze the pool (their freeze_pool.py --final), then the
# full 16-epoch run on our labels (qa/s6r/train.sh) on $CUDA_VISIBLE_DEVICES. No --compile: the smoke runs were
# loader-bound (111 vs 118 windows/s).
set -euo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.tmp/s6r/S6F0
cd $P
L=$(ls -t logs/s6r_anchors_* | head -1)
until grep -q "ANCHORS_DONE\|Traceback" $L; do sleep 60; done
grep -q ANCHORS_DONE $L || { echo "anchor precompute failed: $L"; exit 1; }
grep "anchors SUMMARY" $L | tail -1 | cut -c1-400
(cd $W && python -u freeze_pool.py --final)
N=$(python -c "import json; print(len(json.load(open('$W/data/pool_ok.json'))) - len(json.load(open('$W/data/gold58.json'))))")
echo "pool: $N training studies"
POOL=$W/data/pool_ok.json EXPECT=$N OUT=$P/.tmp/s6r/runs/full bash qa/s6r/train.sh
