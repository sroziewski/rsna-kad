#!/usr/bin/env bash
# Deep per-label head (per_label_xfmr, 4 layers) on GPU 1: waits for the EXACT screen name ztta to
# exit, runs a 200-study / 2-epoch smoke as a GATE (memory, throughput, train loss only - smoke AUC
# has a +-0.04 noise floor and is never read), then all 5 folds and the OOF assembly.
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_screen.sh xfmr_q bash qa/stage11/queue_xfmr4.sh
set -u
cd "$(dirname "$0")/../.."
P=/raid/sroziewski/sandbox/bin/python
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/main_armn_xfmr4_v3p_384.yaml
RUN=main_armn_xfmr4_v3p_384
echo "waiting for ztta since $(date -Is)"
while screen -ls | grep -qE '[0-9]+[.]ztta[[:space:]]'; do sleep 120; done
echo "ztta gone at $(date -Is)"

echo "=== $(date -Is) smoke"
$P scripts/train.py --config $CFG --fold 0 --limit 200 --epochs 2 --run-id armn_smoke_xfmr4_v3p \
  --workers 8 --no-resume || { echo "SMOKE CRASHED"; exit 1; }
rm -f runs/armn_smoke_xfmr4_v3p/fold0/last.pt
# Gate: the head must not blow up memory (arm C recipe peaks at 14.0 GB; 30 GB is our budget on a
# shared GPU), must keep throughput (arm C ~300 img/s), and the loss must be finite and falling.
$P - <<'PY' || { echo "SMOKE GATE FAILED - not launching the 5 folds"; exit 1; }
import json, math, sys
h = json.load(open("runs/armn_smoke_xfmr4_v3p/fold0/metrics.json"))["history"]
e0, e1 = h[0], h[-1]
print(f"smoke: loss {e0['train_loss']:.4f} -> {e1['train_loss']:.4f}, "
      f"{e1['img_per_s']:.0f} img/s, peak {max(r['peak_mem_gb'] for r in h):.1f} GB")
ok = (all(math.isfinite(r["train_loss"]) for r in h) and e1["train_loss"] < e0["train_loss"]
      and max(r["peak_mem_gb"] for r in h) <= 30.0 and e1["img_per_s"] >= 200)
sys.exit(0 if ok else 1)
PY
echo "SMOKE PASSED $(date -Is)"

for k in 0 1 2 3 4; do
  echo "=== $(date -Is) $RUN fold $k"
  $P scripts/train.py --config $CFG --fold $k || exit 1
  rm -f runs/$RUN/fold$k/last.pt   # /raid at ~91 %
done
$P scripts/predict_oof.py --config $CFG || exit 1
echo "XFMRDONE $(date -Is)"
