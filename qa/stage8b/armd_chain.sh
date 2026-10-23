#!/usr/bin/env bash
# Arm D 5-fold chain, modelled on the arm C chain: GPU 1 is shared with a job that has taken
# ~37 GB before, so every fold waits for memory first and a fold killed by OOM is retried
# (scripts/train.py resumes from last.pt). last.pt is deleted at fold end - /raid is at 92 %.
# The OOF is re-assembled after every fold so a partial chain is still comparable.
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_screen.sh armD bash qa/stage8b/armd_chain.sh
set -u
PY=/raid/sroziewski/sandbox/bin/python
CFG=${CFG:-configs/main_armd_convnextv2_t_384.yaml}
RUN=${RUN:-main_armd_convnextv2_t_384}
need=${NEED_MIB:-22000}

wait_gpu() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU:-1}")
    if [ "${f:-0}" -ge "$need" ]; then echo "$(date -Is) GPU${GPU:-1} free ${f} MiB"; return; fi
    echo "$(date -Is) waiting for GPU${GPU:-1}: ${f} MiB free < ${need} MiB"
    sleep 120
  done
}

for k in ${FOLDS:-0 1 2 3 4}; do
  for try in 1 2 3; do
    wait_gpu
    echo "===== fold $k try $try $(date -Is)"
    if $PY scripts/train.py --config "$CFG" --fold "$k"; then break; fi
    echo "FAILED fold $k try $try $(date -Is)"
    if [ "$try" = 3 ]; then echo "GIVING UP on fold $k"; exit 1; fi
    sleep 300
  done
  rm -f "runs/$RUN/fold$k/last.pt"
  $PY scripts/predict_oof.py --config "$CFG" --folds $(seq -s' ' 0 "$k") || true
done
echo ALLDONE
