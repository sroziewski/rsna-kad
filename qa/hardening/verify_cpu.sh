#!/usr/bin/env bash
# The hardened notebook (release/v7 HEAD) must reproduce the previous notebook's output exactly when
# nothing hangs: CPU fp32, 8 threads, the 3 public studies, against the reference CSVs written by the
# earlier dry runs (same threads):
#   main_v5_C          qa/main_v5c_crop/dryrun/submission_main_v5_C.csv   (notebook d8457bf)
#   main_v5_C_silentq  qa/main_v5c_silentq/dryrun/submission.csv          (notebook e935d0b)
#   main_v5_C12        qa/main_v5c12/dryrun/submission_main_v5_C12.csv    (notebook d8457bf)
#   bash qa/hardening/verify_cpu.sh
set -u
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.worktrees/release_v7
PY=/raid/sroziewski/sandbox/bin/python
cd "$P"
OUT=qa/hardening/verify
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/.tmp CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
renice -n 10 $$ > /dev/null
echo "notebook $(md5sum < "$W/notebooks/kaggle_inference.py" | cut -c1-32) ($(git -C "$W" log --oneline -1))"
for pair in "main_v5_C qa/main_v5c_crop/dryrun/submission_main_v5_C.csv" \
            "main_v5_C_silentq qa/main_v5c_silentq/dryrun/submission.csv" \
            "main_v5_C12 qa/main_v5c12/dryrun/submission_main_v5_C12.csv"; do
  set -- $pair
  mkdir -p "$OUT/$1"
  $PY "$W/notebooks/kaggle_inference.py" --data-dir /raid/sroziewski/data/rsna_kad --weights "export/$1" \
    --out "$OUT/$1/submission.csv" --device cpu --fp32 > "$OUT/$1/notebook.log" 2>&1
  rc=$?
  $PY - "$OUT/$1" "$2" "$rc" <<'EOF'
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
d, ref, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
L = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
rd = lambda p: pd.read_csv(p).sort_values("StudyInstanceUID").reset_index(drop=True)
a, b = rd(d / "submission.csv"), rd(ref)
diff = np.abs(a[L].to_numpy(np.float64) - b[L].to_numpy(np.float64)).max()
log = (d / "notebook.log").read_text()
rep = json.loads((d / "run_report.json").read_text())
ok = (rc == 0 and list(a.StudyInstanceUID) == list(b.StudyInstanceUID) and diff == 0.0 and "watchdog armed" in log
      and "WATCHDOG FIRED" not in log and re.search(r"\| 0 prior rows", log) is not None and rep["watchdog"] is None)
print(f"{'PASS' if ok else 'FAIL'}  {d.name}: rc {rc}, max abs diff vs previous notebook {diff:.2e}, "
      f"watchdog armed={'watchdog armed' in log} fired={'WATCHDOG FIRED' in log}, report watchdog={rep['watchdog']}, "
      f"prior rows {rep['prior_rows']}")
EOF
done
echo VERIFYDONE
