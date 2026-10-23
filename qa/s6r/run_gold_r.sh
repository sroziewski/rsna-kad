#!/usr/bin/env bash
# s6r judge: gold-58 inference with our checkpoint, then the joint3 + S6 gold blend for ours and for nartaa's checkpoint.
set -euo pipefail
cd /raid/sroziewski/sandbox/rsna_kad
PY=/raid/sroziewski/sandbox/bin/python
GPU=${GPU:-0} bash qa/s6r/e2e_leg_s6r.sh
tail -3 .tmp/s6r/leg_s6r_gold.log
$PY -I qa/s6r/gold_blend.py .tmp/s6r/gold_s6r.csv ours_s6r
$PY -I qa/s6r/gold_blend.py .tmp/s6r/gold_s6fall.csv nartaa
