#!/usr/bin/env bash
# Waits for the EXACT screen name armn_q (the v3p triage chain) to exit, then - only if the triage
# log says TRIAGEDONE - picks the head for the 5-fold run by the same rule the no-prior picker
# uses (the run's own weak AUC, here at the selected last epoch of fold 0; gold has only ~11 rows
# in fold 0 and cannot rank heads) and runs full_chain_v3p.sh on GPU 0.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_screen.sh armn_full bash qa/stage11/queue_full_v3p.sh
set -u
cd "$(dirname "$0")/../.."
echo "waiting for armn_q since $(date -Is)"
while screen -ls | grep -qE '[0-9]+[.]armn_q[[:space:]]'; do sleep 120; done
qlog=$(ls -t logs/armn_q_*.log | head -1)
grep -q TRIAGEDONE "$qlog" || { echo "triage did not finish cleanly ($qlog) - not launching"; exit 1; }
best=$(/raid/sroziewski/sandbox/bin/python - <<'PY'
import json
s = {}
for v in ("residual", "perlabel", "noprior"):
    s[v] = json.load(open(f"runs/main_armn_{v}_v3p_384/fold0/metrics.json"))["oof_weak_auc"]
print(" ".join(f"{k}={v:.4f}" for k, v in s.items()), file=__import__("sys").stderr)
print(max(s, key=s.get))
PY
) || exit 1
echo "fold-0 winner: $best ($(date -Is))"
ARMN_BEST=$best bash qa/stage11/full_chain_v3p.sh
