#!/usr/bin/env bash
# targets v1.1 GPU chain, queued behind whatever holds GPU 0 (armA8 -> stage9 on 2026-09-21).
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_screen.sh tv11 bash scripts/targets_v1_1_chain.sh
#
# Never launch it directly: run_screen.sh is what keeps HOME/XDG and the vLLM caches inside the
# project (qa/stage5 incident - a bare vLLM run polluted /home/sroziewski once).
#
# Step 1 (PATH 1)  a second local extractor family on the fixed 481-study A/B subset, one
#                  candidate after the other, each with a wall-clock budget so a checkpoint that
#                  will not load cannot eat the night. Weights are deleted again at the end:
#                  /raid is at 92 %.
# Step 2 (payoff)  arm C fold 0 retrained on targets v1.1 and scored against the recorded fold-0
#                  v1 run on the SAME arbiters (§4.2: consensus and gold decide, not weak-AUC).
#
# Step 2 does not depend on step 1 and runs even if every candidate fails.
set -u
cd /raid/sroziewski/sandbox/rsna_kad || exit 1

PYLLM=.venv-llm/bin/python
PY=/raid/sroziewski/sandbox/bin/python
SUBSET_BUDGET_S=${SUBSET_BUDGET_S:-5400}      # per candidate: download + load + 481 reports
BASE_TABLE=data/labels/llm_v1.parquet
PUBLIC_CSV=qa/competitor/llm_labels/report_labels_v2.csv
CANDIDATES=("gemma3:configs/llm_gemma3_27b_w4a16.yaml"
            "mistral24b:configs/llm_mistral_small_24b_w4a16.yaml")

cfg_of() { for s in "${CANDIDATES[@]}"; do [ "${s%%:*}" = "$1" ] && { echo "${s#*:}"; return; }; done; }

wait_gpu0() {
  echo "$(date -Is) waiting for GPU 0 (screens: $*) ..."
  while :; do
    busy=0
    for s in "$@"; do screen -ls | grep -q "[.]${s}[[:space:]]" && busy=1; done
    [ "$busy" = 0 ] && break
    sleep 300
  done
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    [ "${used:-99999}" -lt 8000 ] && break
    sleep 60
  done
  echo "$(date -Is) GPU 0 free (${used} MiB used)"
}

# ----------------------------------------------------------------- step 1: PATH 1
wait_gpu0 armA8 stage9
ok_tags=()
for spec in "${CANDIDATES[@]}"; do
  tag=${spec%%:*}; cfg=${spec#*:}
  echo "===== PATH 1 candidate $tag ($cfg) $(date -Is)"
  if timeout "$SUBSET_BUDGET_S" $PYLLM qa/llm_ab/run_subset.py --config "$cfg" --variant v1 \
        --tag "$tag" --out-dir data/labels; then
    ok_tags+=("$tag")
    echo "===== $tag OK $(date -Is)"
  else
    echo "===== $tag FAILED (exit $?) $(date -Is); moving on"
  fi
  # A second candidate only earns its GPU hour if the first one did not already clear the bar;
  # the comparison below is what says so, so both are run and the report decides.
done

if [ ${#ok_tags[@]} -gt 0 ]; then
  args=()
  cmp_args=(--table "qwen_v1=$BASE_TABLE")
  for t in "${ok_tags[@]}"; do
    args+=(--voter "$t=data/labels/llm_bf16_${t}.parquet")
    cmp_args+=(--table "$t=data/labels/llm_bf16_${t}.parquet")
  done
  args+=(--voter "public=$PUBLIC_CSV")
  $PY qa/llm_ab/voter_gain.py --base "$BASE_TABLE" "${args[@]}" --out qa/llm_ab/path1_voter_gain \
    || echo "voter_gain failed"
  $PY qa/llm_ab/compare.py "${cmp_args[@]}" --baseline qwen_v1 --out qa/llm_ab/path1_comparison \
    || echo "compare failed"
fi

# ----------------------------------------------------------------- step 2: the payoff
echo "===== arm C fold 0 on targets v1.1 $(date -Is)"
wait_gpu0 armA8 stage9
$PY scripts/train.py --config configs/main_armc_v1_1_f0.yaml --fold 0 || {
  echo "arm C v1.1 fold 0 FAILED"; exit 1; }
rm -f runs/main_armc_v1_1_f0/fold0/last.pt

# evaluate_runs keys candidates on the directory name, so the two fold-0 OOF tables are staged
# under distinct names. Nothing is written into either training run.
mkdir -p qa/stage10/armc_v1 qa/stage10/armc_v1_1
cp runs/main_armc_convnextv2_t_384/fold0/oof.parquet qa/stage10/armc_v1/oof.parquet
cp runs/main_armc_v1_1_f0/fold0/oof.parquet          qa/stage10/armc_v1_1/oof.parquet

# Scored on the v1 arbiters: `consensus`, `strict` and `gold58` do not depend on the target
# version at all, and pinning `weak` to v1 keeps one fixed yardstick for both runs. The second
# call repeats it against the v1.1 table so the asymmetry is visible rather than assumed.
$PY scripts/evaluate_runs.py --runs qa/stage10/armc_v1 qa/stage10/armc_v1_1 \
  --labels data/labels/llm_v1.parquet --targets-dir data/targets/v1 \
  --targets-config configs/targets_v1.yaml --n-boot 2000 \
  --out qa/stage10/armc_v1_1_on_v1_arbiters.md --json-out qa/stage10/armc_v1_1_on_v1_arbiters.json
$PY scripts/evaluate_runs.py --runs qa/stage10/armc_v1 qa/stage10/armc_v1_1 \
  --labels data/labels/merged_v1_1.parquet --targets-dir data/targets/v1_1 \
  --targets-config configs/targets_v1_1.yaml --n-boot 2000 \
  --out qa/stage10/armc_v1_1_on_v11_arbiters.md --json-out qa/stage10/armc_v1_1_on_v11_arbiters.json

# --------------------------------------------- step 3: full corpus, only if the subset earned it
# The gate is §4.2 itself, read off the subset measurement: >= 0.010 with P >= 0.95 on the
# 4-voter merge. Below that the 4,407-report pass is 1.5 GPU-h spent on noise.
winner=$($PY - <<'PY'
import json, pathlib
p = pathlib.Path("qa/llm_ab/path1_voter_gain.json")
if not p.exists():
    print(""); raise SystemExit
d = json.loads(p.read_text())
best, best_mean = "", 0.0
for name, delta in d.get("deltas", {}).items():
    if not name.endswith("_merge_4v") or name.startswith("public_"):
        continue
    if delta["mean"] >= 0.010 and delta["p_gt"] >= 0.95 and delta["mean"] > best_mean:
        best, best_mean = name[:-len("_merge_4v")], delta["mean"]
print(best)
PY
)
if [ -n "$winner" ]; then
  echo "===== PATH 1 winner '$winner' cleared the bar; full corpus $(date -Is)"
  wait_gpu0 armA8 stage9
  cfg=$(cfg_of "$winner")
  if [ -n "$cfg" ] && $PYLLM scripts/extract_labels_llm.py --config "$cfg" run --variant v1 \
       --tag "$winner" --gold; then
    $PY scripts/make_labels_v1_1.py --voter "data/labels/llm_v1_${winner}.parquet" \
        --out data/labels/merged_v1_1_${winner}.parquet
    $PY scripts/make_targets.py --labels data/labels/merged_v1_1_${winner}.parquet \
        --config configs/targets_v1_1.yaml --folds data/folds/folds_v1.csv \
        --out data/targets/v1_1_${winner}
  else
    echo "full-corpus pass FAILED for $winner"
  fi
else
  echo "===== no PATH 1 candidate cleared the §4.2 bar on the subset; full corpus skipped"
fi

# ----------------------------------------------------------------- cleanup
for repo in models--RedHatAI--gemma-3-27b-it-quantized.w4a16 \
            models--RedHatAI--Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16; do
  rm -rf "${HF_HOME:-/raid/sroziewski/cache/huggingface}/hub/$repo"
done
echo "===== chain done $(date -Is)"
du -sh "${HF_HOME:-/raid/sroziewski/cache/huggingface}"
df -h /raid | tail -1
