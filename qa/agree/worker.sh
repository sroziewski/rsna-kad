#!/usr/bin/env bash
# qa/agree/worker.sh <gpu> [lead|<delay s>] (2026-10-14): the lead builds data/targets/v3_pseudo_full_agree and
# smoke-tests the config, then writes qa/agree/INPUTS_OK.  Every worker then claims seeds (atomic mkdir) whenever its
# GPU has >= 18 GB free; the last one to finish runs the judge.  Same flow as qa/oai/worker.sh.
set -u
cd /raid/sroziewski/sandbox/rsna_kad
gpu=$1
P=/raid/sroziewski/sandbox/bin/python
Q=qa/agree
C=$Q/claims
SEEDS="2026 2126 2226 2326 2426 2526"
CFG=configs/main_armc_pseudo_full_agree.yaml
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
mkdir -p $C

wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")" -lt 18000 ]; do sleep 120; done; }
seed_done() { local d=runs/main_armc_convnextv2_t_384_p_full_agree_s$1/full
  [ -f $d/oof.parquet ] && [ -f $d/model.safetensors ] && [ ! -f $d/last.pt ]; }

if [ "${2:-}" = lead ]; then
  { CUDA_VISIBLE_DEVICES= $P $Q/make_targets_agree.py \
    && rm -rf .tmp/agree_smoke \
    && CUDA_VISIBLE_DEVICES=$gpu $P scripts/train.py --config $CFG --fold full --seed 2026 --limit 40 --epochs 1 \
         --runs-dir .tmp/agree_smoke --no-resume \
    && rm -rf .tmp/agree_smoke; } > $Q/inputs.log 2>&1 && touch $Q/INPUTS_OK || { echo "inputs FAILED"; touch $Q/INPUTS_FAILED; }
elif [ -n "${2:-}" ]; then
  sleep "$2"
fi
while [ ! -f $Q/INPUTS_OK ]; do [ -f $Q/INPUTS_FAILED ] && { echo "inputs failed - stop"; exit 1; }; sleep 60; done

for s in $SEEDS; do
  wait_gpu
  mkdir $C/s_$s 2>/dev/null || continue
  echo "$(date -Is) GPU $gpu takes seed $s"
  CUDA_VISIBLE_DEVICES=$gpu $P scripts/train.py --config $CFG --fold full --seed $s \
    && echo "$(date -Is) ALLDONE seed s$s" || echo "$(date -Is) FAILED seed s$s"
  sleep 300
done

all=1; for s in $SEEDS; do seed_done $s || all=0; done
if [ $all = 1 ] && mkdir $C/judge 2>/dev/null; then
  CUDA_VISIBLE_DEVICES= $P $Q/judge.py > $Q/compare.txt 2>&1; touch $Q/DONE
fi
