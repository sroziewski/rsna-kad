#!/usr/bin/env bash
# main_v5_C 12 seeds probe: export the 12 full-data arm-C seeds (equal weights, main_v5_C's prior,
# no TTA, K 24) with release/v7's exporter, then check it on CPU fp32 over the 3 public studies:
#   - the notebook loads 12 members, 0 prior rows
#   - the mean of the first 6 members (main_v5_C's) reproduces main_v5_C's output at the same
#     thread count (qa/main_v5c_crop/dryrun/submission_main_v5_C.csv, release/v7 notebook, 8 threads)
#   - the 12-member output is the mean of the two 6-member halves
#   bash qa/main_v5c12/dryrun.sh
set -u
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.worktrees/release_v7
PY=/raid/sroziewski/sandbox/bin/python
cd "$P"
OUT=qa/main_v5c12/dryrun
mkdir -p "$OUT"
export PYTHONPATH=$W/src PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/.tmp CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8   # same as the main_v5_C reference run
renice -n 10 $$ > /dev/null
R=$(echo runs/main_armc_convnextv2_t_384_p_full_s{2026,2126,2226,2326,2426,2526,2626,2726,2826,2926,3026,3126})
if [ ! -d export/main_v5_C12 ]; then
  # shellcheck disable=SC2086
  $PY "$W/scripts/export_kaggle.py" --name main_v5_C12 --export-root export --wheels-dir export/wheels \
    --user sroziews --dataset-slug rsna-kad-weights-v5c12 --no-tta --k-infer 24 \
    --group "C=${R%% *}" ${R#* } --prior-of-bundle export/main_v5_C > "$OUT/export.log" 2>&1
  echo "export rc=$?"
fi
$PY "$W/notebooks/kaggle_inference.py" --data-dir /raid/sroziewski/data/rsna_kad --weights export/main_v5_C12 \
  --out "$OUT/submission_main_v5_C12.csv" --device cpu --fp32 > "$OUT/notebook.log" 2>&1; echo "notebook rc=$?"
$PY - "$OUT" <<'EOF' 2>&1 | tee "$OUT/checks.txt"
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from rsna_kad.constants import ID_COL, LABELS
from rsna_kad.infer import members_from_ensemble, predict_studies, test_study_dirs
out = Path(sys.argv[1])
fails = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"); ok or fails.append(name)
doc, v5c = (json.loads(Path(f"export/{b}/ensemble.json").read_text()) for b in ("main_v5_C12", "main_v5_C"))
names = [m["path"].split("/")[1] for m in doc["members"]]
check("12 members, the 12 seeds in order", len(names) == 12 and all(f"_s{s}_full" in n for n, s in zip(names, [2026,2126,2226,2326,2426,2526,2626,2726,2826,2926,3026,3126])))
check("first 6 members == main_v5_C's", names[:6] == [m["path"].split("/")[1] for m in v5c["members"]])
check("weights 1/12, no TTA, K 24", all(abs(m["weight"] - 1/12) < 1e-12 and m["tta"] is False and "view_scales" not in m for m in doc["members"]) and doc["K_infer"] == 24)
check("prior == main_v5_C's (56687125)", doc["prior"] == v5c["prior"], doc["prior_source"])
meta = json.loads(Path("export/main_v5_C12/dataset-metadata.json").read_text())
check("dataset CC-BY-NC-4.0, slug v5c12, no NOTICE", meta["licenses"] == [{"name": "CC-BY-NC-4.0"}] and meta["id"] == "sroziews/rsna-kad-weights-v5c12"
      and not Path("export/main_v5_C12/NOTICE").exists())
log = (out / "notebook.log").read_text()
check("notebook: 12 members, mode main, 12 loaded, 0 prior rows, no traceback",
      "12 members, mode main" in log and len(re.findall(r"INFO loaded ", log)) == 12 and re.search(r"\| 0 prior rows", log) is not None and "Traceback" not in log)
rd = lambda p: pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)
sub, ref = rd(out / "submission_main_v5_C12.csv"), rd("qa/main_v5c_crop/dryrun/submission_main_v5_C.csv")
data = Path("/raid/sroziewski/data/rsna_kad")
sample = pd.read_csv(data / "sample_submission.csv", dtype={ID_COL: str})
series = pd.read_csv(data / "test_series.csv", dtype=str)
dirs = test_study_dirs(data, series, sample)
ms = members_from_ensemble("export/main_v5_C12")
half = []
for part in (ms[:6], ms[6:]):
    p = predict_studies(part, dirs, series, devices=["cpu"], workers=3, fp16=False)
    half.append(p.set_index(ID_COL).loc[sub[ID_COL], list(LABELS)].to_numpy(np.float64))
X = lambda d: d[list(LABELS)].to_numpy(np.float64)
d6 = np.abs(half[0] - X(ref)).max()
check("first-6 mean reproduces main_v5_C (same threads; CSV rounds to 1e-6)", d6 <= 5.01e-7, f"max abs {d6:.2e}")
dm = np.abs(X(sub) - 0.5 * (half[0] + half[1])).max()
check("12-member output == mean of the two halves", dm <= 5.01e-7, f"max abs {dm:.2e}")
dv = np.abs(X(sub) - X(ref))
print(f"INFO  12 seeds vs main_v5_C: mean abs {dv.mean():.4f}, max {dv.max():.4f}; halves differ by mean {np.abs(half[0]-half[1]).mean():.4f}")
t = float(re.search(r"inference [0-9.]+s = ([0-9.]+)s/study", log).group(1))
print(f"INFO  CPU fp32 s/study: 12 members {t} vs main_v5_C 105.99 at the same threads (x{t / 105.99:.2f})")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
EOF
echo DRYRUNDONE
