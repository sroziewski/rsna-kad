#!/usr/bin/env bash
# Waits for the EXACT screen name pv4_gpu0 (round-2 pseudo chain: E_p2 then A_p2) to exit, then
# runs the v3p triage on GPU 0 with the CPU-only no-prior head picker alongside it.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_screen.sh armn_q bash qa/stage11/queue_v3p.sh
set -u
cd "$(dirname "$0")/../.."
echo "waiting for pv4_gpu0 since $(date -Is)"
while screen -ls | grep -qE '[0-9]+[.]pv4_gpu0[[:space:]]'; do sleep 120; done
echo "pv4_gpu0 gone at $(date -Is)"
/raid/sroziewski/sandbox/bin/python qa/stage11/pick_noprior_head.py _v3p > logs/armn_pick_v3p.log 2>&1 &
bash qa/stage11/triage_v3p.sh
