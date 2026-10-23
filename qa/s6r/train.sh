#!/usr/bin/env bash
# s6r training: nartaa's production argv (run_all_alldata.sh ARGV) unchanged except the pool / --expect-train (ours),
# the arm (OURS = our soft target L|prior with our cell weight L|w, weight-normalised BCE as their E_prior) and paths.
#   POOL=<pool json> EXPECT=<n> OUT=<dir> [EPOCHS=16] [EXTRA="--stop-after-epoch 0"] bash qa/s6r/train.sh
set -euo pipefail
R=/raid/sroziewski/sandbox/rsna_kad/.tmp/s6r
W=$R/S6F0
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W
python -u train_s6f0_all.py --pool "$POOL" --folds-json $R/folds_v0.json --targets $W/data/targets_s6f0.csv \
  --out "$OUT" --arms "OURS=bce:prior:w" --backbone convnext_small --init $R/init_model.safetensors --seeds 1 --seed-base 42 \
  --folds 5 --only-folds 0 --expect-train "$EXPECT" --expect-holdout 0 --all-data \
  --epochs "${EPOCHS:-16}" --batch 2 --train-windows 64 --size 320 --views v4 --crop 288 --slices 3 --phys-mm 3.5 \
  --spatial-attn --grad-ckpt --lr 1e-4 --head-lr 1e-3 --wd 0.05 --aug crop --train-sampling uniform --ema 0 \
  --workers 8 --loader-timeout 600 --max-fail-frac 0.005 --log-every 50 --work-tmp $R/work_$(basename "$OUT") \
  --resume --no-final-predict --i-have-user-authorisation-to-finetune ${EXTRA:-}
echo S6R_TRAIN_DONE
