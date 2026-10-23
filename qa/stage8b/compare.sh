#!/usr/bin/env bash
# Task 1: the honest comparison of A, E, C (and the 8-epoch arm A variant when its chain has
# landed) and of every ensemble of them, before one submission is spent.
#
# Two tools because they answer different halves of §4.2: compare_ensemble.py blends the runs
# and gives the paired weak/strict/gold deltas per combination and the per-label table;
# evaluate_runs.py adds `consensus`, the companion arbiter the rule makes mandatory, and the
# gold-58 veto with its CI. All runs here are trained on targets v1, so `weak` stays primary
# and `consensus` is the cross-check rather than the other way round.
#
#   bash qa/stage8b/compare.sh            # writes qa/stage8b/compare_*.md
set -u
P=/raid/sroziewski/sandbox/bin/python
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR" qa/stage8b

A=runs/main_convnext_s_384
E=runs/eff_effv2s_256
C=runs/main_armc_convnextv2_t_384
A8=runs/main_convnext_s_384_e8

# train_fold rewrites fold<k>/{oof.parquet,metrics.json} after EVERY epoch, so a fold that is
# still running contributes half-trained predictions to the assembled OOF and silently drags the
# arm down (measured: arm C's fold 1 read weak 0.9054 at epoch 2 against 0.9282 at epoch 9 on
# fold 0). Nothing downstream can detect that, so refuse to compare until the chain says ALLDONE.
for pair in "armC:$C" "armA8:$A8"; do
  name=${pair%%:*}; dir=${pair#*:}
  [ -d "$dir" ] || continue
  if screen -ls | grep -q "[.]${name}[[:space:]]"; then
    echo "$name is still running - refusing to score $dir" >&2
    [ "$name" = armC ] && exit 1
    A8=/nonexistent
  fi
done

runs="A=$A E=$E C=$C"
combos="A E C A+C A+E A+E+C"
if [ -f "$A8/oof.parquet" ]; then
  runs="$runs A8=$A8"
  combos="$combos A8 A8+E+C"
fi

echo "### 1. arms and ensembles (paired, 2000 replicates)"
# shellcheck disable=SC2086
$P scripts/compare_ensemble.py --runs $runs --combos $combos --n-boot 2000 \
  | tee qa/stage8b/compare_combos.md

echo
echo "### 2. arbiters incl. consensus"
args="--runs $A $E $C"
[ -f "$A8/oof.parquet" ] && args="$args $A8"
for blend in "main_convnext_s_384 eff_effv2s_256 main_armc_convnextv2_t_384" \
             "main_convnext_s_384 eff_effv2s_256" \
             "main_convnext_s_384 main_armc_convnextv2_t_384"; do
  tag=$(echo "$blend" | tr ' ' '+')
  echo "--- blend: $tag"
  # shellcheck disable=SC2086
  $P scripts/evaluate_runs.py $args --blend --blend-of $blend --n-boot 2000 \
    --out "qa/stage8b/arbiters_$(echo "$blend" | tr ' ' '_' | md5sum | cut -c1-8).md" \
    | sed -n '/## 2. Scores/,/## 4./p'
done
echo COMPAREDONE
