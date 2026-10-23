#!/usr/bin/env bash
# Arm D candidate sweep: 200 studies x 2 epochs per candidate on fold 0 (architecture §4.1,
# the arm C method). Decided on epoch-1 train loss, throughput and peak memory - NOT on the
# smoke AUC, whose noise floor at this size is +-0.04.
#
# Gate to beat: arm A's own smoke runs/main_smoke_384 epoch 1 = loss 0.7091, 161 img/s, 15.7 GB.
# Arm C's recorded control armc_sweep_c1 = 0.7091, 210 img/s, 11.3 GB; d0 re-measures it here.
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_screen.sh armD_sweep bash qa/stage8b/sweep.sh
set -u
PY=/raid/sroziewski/sandbox/bin/python
need=${NEED_MIB:-20000}

wait_gpu() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU:-1}")
    if [ "${f:-0}" -ge "$need" ]; then echo "$(date -Is) GPU${GPU:-1} free ${f} MiB"; return; fi
    echo "$(date -Is) waiting for GPU${GPU:-1}: ${f} MiB free < ${need} MiB"
    sleep 60
  done
}

for c in ${CANDIDATES:-d0 d1 d2 d3}; do
  wait_gpu
  echo "===== $c $(date -Is)"
  $PY scripts/train.py --config "configs/main_armd_sweep_${c}.yaml" --fold 0 --limit 200 --no-resume \
    || echo "FAILED $c"
  rm -f "runs/armd_sweep_${c}/fold0/last.pt"
done

echo "===== summary"
printf '| candidate | views | series | train loss @e1 | img/s | peak GB | train s | eval s |\n'
printf '|---|---|---|---|---|---|---|---|\n'
for c in ${CANDIDATES:-d0 d1 d2 d3}; do
  log="runs/armd_sweep_${c}/fold0/train.log"
  [ -f "$log" ] || continue
  $PY - "$c" "$log" <<'EOF'
import re, sys, yaml
c, log = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(f"configs/main_armd_sweep_{c}.yaml"))["model"]
slots = cfg["slots"]
line = [l for l in open(log) if "epoch 1 done" in l]
m = re.search(r"loss ([\d.]+).*train (\d+)s eval (\d+)s (\d+) img/s peak ([\d.]+) GB", line[-1]) if line else None
print(f"| {c} | {len(slots)} | {len(set(slots))} | " +
      (f"{m.group(1)} | {m.group(4)} | {m.group(5)} | {m.group(2)} | {m.group(3)} |" if m else "- | - | - | - | - |"))
EOF
done
echo SWEEPDONE
