#!/usr/bin/env bash
# release/v7 dry run of the merged A+B path (nothing is pushed):
#   A = main_v5_C's 6 Cfull members, B = one pipeline-B run, group weights 0.5/0.5, prior from
#   Cp's 5-fold OOF, exported with release/v7's export_kaggle.py and run through release/v7's
#   notebook on CPU fp32 over the 3 public studies. A B-only bundle is run as well, so
#   check_dryrun.py can prove the merge arithmetic (blend == 0.5 x main_v5_C + 0.5 x B).
#   Both provisional bundles are deleted at the end; the evidence stays in qa/release_v7/dryrun_<tag>/.
#
#   bash qa/release_v7/dryrun.sh <B run dir> <tag> [wait-for-file]
# With wait-for-file (runs/b_gate/G0.json) it first waits, polling on the server, until that file
# exists - B's gate-0 chain writes it after the smoke and its benchmark have finished.
set -u
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.worktrees/release_v7
PY=/raid/sroziewski/sandbox/bin/python
B=$1 TAG=$2 WAIT=${3:-}
cd "$P"
if [ -n "$WAIT" ]; then
  for i in $(seq 1 288); do [ -f "$WAIT" ] && break; sleep 300; done
  [ -f "$WAIT" ] || { echo "gave up waiting for $WAIT after 24 h"; exit 1; }
  echo "$(date -Is) $WAIT present: $(cat "$WAIT" | head -c 300)"
fi
OUT=qa/release_v7/dryrun_$TAG
mkdir -p "$OUT"
export PYTHONPATH=$W/src PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/.tmp CUDA_VISIBLE_DEVICES=
# B's training chains share this machine's CPUs with their dataloaders: cap and nice the CPU inference.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
renice -n 10 $$ > /dev/null
A=$(echo runs/main_armc_convnextv2_t_384_p_full_s{2026,2126,2226,2326,2426,2526})
COMMON="--export-root export --wheels-dir export/wheels --user sroziews --dataset-slug rsna-kad-weights-dryrun-v7
  --no-tta --allow-smoke --folds 0 --prior-from runs/main_armc_convnextv2_t_384_p"
# shellcheck disable=SC2086
$PY "$W/scripts/export_kaggle.py" --name _dryrun_v7 $COMMON --group "A=${A%% *}" ${A#* } --group "B=$B" \
  --group-weights A=0.5,B=0.5 > "$OUT/export.log" 2>&1; echo "export A+B rc=$?"
# shellcheck disable=SC2086
$PY "$W/scripts/export_kaggle.py" --name _dryrun_v7_b $COMMON --run "$B" > "$OUT/export_b.log" 2>&1; echo "export B rc=$?"
for b in _dryrun_v7 _dryrun_v7_b; do
  [ -f "export/$b/ensemble.json" ] || { echo "no export/$b"; exit 1; }
  $PY "$W/notebooks/kaggle_inference.py" --data-dir /raid/sroziewski/data/rsna_kad --weights "export/$b" \
    --out "$OUT/submission$b.csv" --device cpu --fp32 > "$OUT/notebook$b.log" 2>&1; echo "notebook $b rc=$?"
done
$PY qa/release_v7/check_dryrun.py export/_dryrun_v7 "$OUT" 2>&1 | tee "$OUT/checks.txt"; echo "checks rc=${PIPESTATUS[0]}"
for f in ensemble.json NOTICE dataset-metadata.json VERSION; do cp "export/_dryrun_v7/$f" "$OUT/" 2>/dev/null; done
md5sum "export/_dryrun_v7/LICENSE-DINOv3.md" > "$OUT/license_md5.txt" 2>/dev/null
rm -rf export/_dryrun_v7 export/_dryrun_v7_b && echo "provisional bundles deleted"
git -C "$W" status --short | sed 's/^/worktree: /'
echo DRYRUNDONE
