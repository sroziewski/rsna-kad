#!/usr/bin/env bash
# Student (eff_pseudo_256, targets v3_pseudo) vs its same-recipe baseline (eff_effv2s_256,
# targets v1), on every arbiter, paired bootstrap. gold58 and consensus are the yardsticks;
# weak is printed on the v1 and v1_1 tables but is CONTAMINATED for the student (its training
# targets carry teacher OOF from models that trained on each fold's weak rows).
#   bash qa/pseudo_v3/compare.sh
set -eu
P=/raid/sroziewski/sandbox/bin/python
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" CUDA_VISIBLE_DEVICES=
if screen -ls | grep -q "[.]pseudo[[:space:]]"; then
  echo "screen 'pseudo' is still running - refusing to score a live chain" >&2; exit 1
fi
B=runs/eff_effv2s_256; S=runs/eff_pseudo_256
[ "$(ls $S/fold*/model.safetensors | wc -l)" = 5 ] || { echo "student has < 5 finished folds" >&2; exit 1; }
$P scripts/evaluate_runs.py --runs $B $S --n-boot 2000 \
  --labels data/labels/llm_v1.parquet --targets-dir data/targets/v1 --targets-config configs/targets_v1.yaml \
  --out qa/pseudo_v3/arbiters_v1.md --json-out qa/pseudo_v3/arbiters_v1.json
$P scripts/evaluate_runs.py --runs $B $S --n-boot 2000 --arbiters weak strict consensus gold58 \
  --labels data/labels/merged_v1_1.parquet --targets-dir data/targets/v1_1 --targets-config configs/targets_v1_1.yaml \
  --out qa/pseudo_v3/arbiters_v1_1.md --json-out qa/pseudo_v3/arbiters_v1_1.json
