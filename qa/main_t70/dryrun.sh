#!/usr/bin/env bash
# Probe "t70 heavier teacher 3 seeds": the 3 full-data arm-C models trained on v3_pseudo_full_t70, 1/3 each,
# main_v5_C's prior, no TTA, K 24, CC-BY-NC-4.0, hardened notebook (release/v7 a3c5cc1). CPU fp32, 8 threads:
#   bash qa/main_t70/dryrun.sh
set -u
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.worktrees/release_v7
PY=/raid/sroziewski/sandbox/bin/python
cd "$P"
OUT=qa/main_t70/dryrun
mkdir -p "$OUT"
export PYTHONPATH=$W/src PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/.tmp CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
renice -n 10 $$ > /dev/null
T=$(echo runs/main_armc_convnextv2_t_384_p_full_t70_s{2026,2126,2226})
if [ ! -d export/main_t70 ]; then
  # shellcheck disable=SC2086
  $PY "$W/scripts/export_kaggle.py" --name main_t70 --export-root export --wheels-dir export/wheels \
    --user sroziews --dataset-slug rsna-kad-weights-t70 --no-tta --k-infer 24 \
    --group "T=${T%% *}" ${T#* } --prior-of-bundle export/main_v5_C > "$OUT/export.log" 2>&1
  echo "export rc=$?"
fi
[ -f export/main_t70/ensemble.json ] || { echo "no export"; exit 1; }
$PY "$W/notebooks/kaggle_inference.py" --data-dir /raid/sroziewski/data/rsna_kad --weights export/main_t70 \
  --out "$OUT/submission.csv" --device cpu --fp32 > "$OUT/notebook.log" 2>&1; echo "notebook rc=$?"
$PY - "$OUT" <<'EOF' 2>&1 | tee "$OUT/checks.txt"
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from rsna_kad.constants import ID_COL, LABELS
from rsna_kad.infer import members_from_ensemble, predict_studies, test_study_dirs
out = Path(sys.argv[1]); bundle = Path("export/main_t70")
fails = []
def check(n, ok, d=""):
    print(f"{'PASS' if ok else 'FAIL'}  {n}  {d}"); ok or fails.append(n)
doc, v5c = json.loads((bundle / "ensemble.json").read_text()), json.loads(Path("export/main_v5_C/ensemble.json").read_text())
names = [m["path"].split("/")[1] for m in doc["members"]]
check("3 t70 members", names == [f"main_armc_convnextv2_t_384_p_full_t70_s{s}_full" for s in (2026, 2126, 2226)])
check("weights 1/3, no TTA, K 24, no label weights", all(abs(m["weight"] - 1/3) < 1e-12 and m["tta"] is False and "label_weights" not in m
                                                        for m in doc["members"]) and doc["K_infer"] == 24)
check("prior == main_v5_C's", doc["prior"] == v5c["prior"], doc["prior_source"])
meta = json.loads((bundle / "dataset-metadata.json").read_text())
check("licence CC-BY-NC-4.0, slug t70, no NOTICE", doc["license"] == "CC-BY-NC-4.0" and meta["licenses"] == [{"name": "CC-BY-NC-4.0"}]
      and meta["id"] == "sroziews/rsna-kad-weights-t70" and not (bundle / "NOTICE").exists())
log = (out / "notebook.log").read_text(); rep = json.loads((out / "run_report.json").read_text())
check("notebook: 3 members loaded, 0 prior rows, no traceback", "3 members, mode main" in log and len(re.findall(r"INFO loaded ", log)) == 3
      and re.search(r"\| 0 prior rows", log) is not None and "Traceback" not in log)
check("watchdog armed and silent", "watchdog armed" in log and "WATCHDOG FIRED" not in log and rep["watchdog"] is None)
rd = lambda p: pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)
sub = rd(out / "submission.csv"); ids = list(sub[ID_COL])
data = Path("/raid/sroziewski/data/rsna_kad")
sample = pd.read_csv(data / "sample_submission.csv", dtype={ID_COL: str}); series = pd.read_csv(data / "test_series.csv", dtype=str)
dirs = test_study_dirs(data, series, sample)
run = lambda ms: predict_studies(ms, dirs, series, devices=["cpu"], workers=3, fp16=False).set_index(ID_COL).loc[ids, list(LABELS)].to_numpy(np.float64)
each = [run([m]) for m in members_from_ensemble(bundle)]
full = run(members_from_ensemble(bundle))
d = np.abs(full - np.mean(each, axis=0)).max()
check("full precision: output == mean of the 3 members", d < 1e-12, f"max abs {d:.2e}")
dc = np.abs(sub[list(LABELS)].to_numpy(np.float64) - full).max()
check("notebook CSV == the ensemble at full precision (float32 + 6 decimals)", dc <= 5.2e-7, f"max abs {dc:.2e}")
ref = rd("qa/main_v5c_crop/dryrun/submission_main_v5_C.csv")
dv = np.abs(sub[list(LABELS)].to_numpy(np.float64) - ref[list(LABELS)].to_numpy(np.float64))
print(f"INFO  vs main_v5_C local: mean abs {dv.mean():.4f} max {dv.max():.4f}")
t = float(re.search(r"inference [0-9.]+s = ([0-9.]+)s/study", log).group(1))
print(f"INFO  CPU fp32 s/study {t} vs main_v5_C 105.99 (x{t / 105.99:.2f}) -> hidden run ~{1.2 * t / 105.99:.1f} h")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
EOF
echo DRYRUNDONE
