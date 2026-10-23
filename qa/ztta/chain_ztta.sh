#!/usr/bin/env bash
# Screen 'ztta' (coordinator 2026-09-26): wait for the exact screen 'pv4_gpu1' (C_p2) to exit, then recompute
# the main_v4 members' OOF with/without z-TTA on GPU 1. Nobody waits on 'ztta'.
set -u
cd "$(dirname "$0")/../.."
while screen -ls | grep -q "[.]pv4_gpu1[[:space:]]"; do sleep 120; done
echo "$(date -Is) pv4_gpu1 gone; starting z-TTA OOF on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"
set -e
python qa/ztta/ztta_oof.py
python qa/ztta/ztta_oof.py --check
echo "$(date -Is) ALLDONE ztta"
