#!/usr/bin/env bash
# v3p scoring: PRIMARY = paired gold-58 vs C_p and the blend contribution to main_v4; weak/strict/
# consensus are secondary (pseudo-label targets are built from the teachers' OOF, so they are
# contaminated). Screen guard as in compare.sh (§4.1 trap). Fold 0 holds only ~11 gold rows, so a
# fold-0 gold number is NOT a triage signal - use `full`.
#   BLEND="runs/<A_p> runs/<E_p> runs/main_armc_convnextv2_t_384_p" bash qa/stage11/compare_v3p.sh fold0|full
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp"
mode=${1:-fold0}
alive() { screen -ls | grep -qE "[0-9]+[.]$1[[:space:]]"; }
# Screen guard, per run (the §4.1 trap: train_fold rewrites fold<k>/oof.parquet every epoch): a run
# is skipped while the chain that writes it is alive. The depth-1 heads' fold 0 is final once
# armn_q (the triage) has exited, even while armn_full trains folds 1-4 of the winner.
live() {
  case $1 in
    xfmr4) alive xfmr_q ;;
    *) alive armn_q || { [ "$mode" = full ] && alive armn_full; } ;;
  esac
}
CP=runs/main_armc_convnextv2_t_384_p
have=""
for v in residual perlabel noprior xfmr4; do
  d=runs/main_armn_${v}_v3p_384; [ -d "$d" ] || continue
  if live "$v"; then echo "skipping $d: its chain is still running" >&2; continue; fi
  [ "$mode" = fold0 ] && [ "$v" = xfmr4 ] && continue   # xfmr4 has no triage; judged on 5 folds only
  $P scripts/predict_oof.py --config configs/main_armn_${v}_v3p_384.yaml $( [ "$mode" = fold0 ] && echo "--folds 0" ) >/dev/null || exit 1
  have="$have $d"
done
# evaluate_runs intersects studies across runs, so with fold-0-only arm-N runs every number is fold 0.
[ -n "$have" ] || { echo "nothing scoreable yet" >&2; exit 1; }
$P scripts/evaluate_runs.py --runs $have $CP --n-boot 2000 --out qa/stage11/v3p_${mode}.md | sed -n '/## 2./,/## 5./p'
if [ "$mode" = full ] && [ -n "${BLEND:-}" ]; then
  for d in $have; do
    b=$(for r in $BLEND; do basename "$r"; done | tr '\n' ' ')
    $P scripts/evaluate_runs.py --runs $BLEND $d --blend --blend-of $b --n-boot 2000 \
      --out "qa/stage11/v3p_blend_base.md" >/dev/null
    $P scripts/evaluate_runs.py --runs $BLEND $d --blend --blend-of $b $(basename "$d") --n-boot 2000 \
      --out "qa/stage11/v3p_blend_$(basename "$d").md" | sed -n '/## 2./,/## 4./p'
  done
fi
