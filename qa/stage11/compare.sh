#!/usr/bin/env bash
# Stage 11 / A2 scoring. Two modes:
#   bash qa/stage11/compare.sh fold0     # triage: the fold-0 rows only, paired
#   bash qa/stage11/compare.sh full      # the 5-fold decision, all four arbiters + ensembles
#
# SCREEN GUARD (the trap recorded in architecture §4.1 on 2026-09-21): train_fold rewrites
# fold<k>/oof.parquet after EVERY epoch, so scoring a live chain silently mixes finished folds
# with half-trained ones and nothing downstream notices. Refuse while any armn screen is alive -
# the same rule qa/stage8b/compare.sh applies to armC/armA8.
#
# Every candidate is scored on the **v1 yardstick** (targets v1 + labels llm_v1), exactly as
# §4.3 scored arm A/A8/C, even though the arm-N runs are TRAINED on v1_1_gemma3: `weak` is
# defined by the target table, so a shared yardstick is the only way the numbers compare. The
# arm-N control (main_armn_base_384) is trained on the same targets as the head variants, so the
# head delta is clean; the arm C comparison also carries the schedule and target-table change.
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp"
mkdir -p "$TMPDIR" qa/stage11
mode=${1:-fold0}

if screen -ls | grep -qE '[.]armn[a-z0-9_]*[[:space:]]'; then
  echo "an armn chain is still running - refusing to score" >&2
  screen -ls | grep -E '[.]armn' >&2
  exit 1
fi

N0=runs/main_armn_base_384
N1=runs/main_armn_residual_384
N2=runs/main_armn_perlabel_384
N3=runs/main_armn_noprior_384
C=runs/main_armc_convnextv2_t_384
A=runs/main_convnext_s_384
E=runs/eff_effv2s_256

have="" pairs=""
for d in $N0 $N1 $N2 $N3; do
  [ -d "$d" ] || continue
  tag=N_$(basename "$d" | sed 's/^main_armn_//; s/_384$//')
  have="$have $d"; pairs="$pairs $tag=$d"
done
[ -n "$have" ] || { echo "no arm-N run directories yet" >&2; exit 1; }

folds_arg=""
[ "$mode" = fold0 ] && folds_arg="--folds 0"

# Root oof.parquet from the folds that exist (assemble_oof); harmless to redo.
for d in $have; do
  cfgf=$(ls configs/main_armn_$(basename "$d" | sed 's/^main_armn_//; s/_384$//')_384.yaml)
  # shellcheck disable=SC2086
  $P scripts/predict_oof.py --config "$cfgf" $( [ "$mode" = fold0 ] && echo "--folds 0" ) >/dev/null || exit 1
done

echo "### 1. paired weak / strict / gold on the shared rows ($mode)"
# shellcheck disable=SC2086
$P scripts/compare_ensemble.py --runs $pairs A=$A E=$E C=$C \
  --combos $(for p in $pairs; do echo "${p%%=*}"; done | tr '\n' ' ') A E C \
  $folds_arg --n-boot 2000 | tee "qa/stage11/compare_${mode}.md"

echo
echo "### 2. arbiters incl. consensus (weak / strict / consensus / gold58)"
# shellcheck disable=SC2086
$P scripts/evaluate_runs.py --runs $have $C $A $E --n-boot 2000 \
  --out "qa/stage11/arbiters_${mode}.md" --json-out "qa/stage11/arbiters_${mode}.json" \
  | sed -n '/## 1./,$p'

if [ "$mode" = full ]; then
  echo
  echo "### 3. ensemble contribution (is it worth a member slot?)"
  best=${ARMN_BEST:-N_residual}
  # shellcheck disable=SC2086
  $P scripts/compare_ensemble.py --runs $pairs A=$A E=$E C=$C \
    --combos A+E+C "A+E+C+$best" "A+E+$best" "$best" C --n-boot 2000 \
    | tee qa/stage11/ensemble_full.md
fi
echo COMPAREDONE
