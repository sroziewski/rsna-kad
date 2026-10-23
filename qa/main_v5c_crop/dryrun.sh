#!/usr/bin/env bash
# main_v5_C + crop TTA probe: derive the bundle from export/main_v5_C (only the views change) and
# run release/v7's notebook on CPU fp32 over the 3 public studies, three ways:
#   base  = export/main_v5_C as shipped          -> must equal main_v5_C's local output (<= 5e-7)
#   crop  = export/main_v5_C_crop (1.0 + 115/130) -> the probe
#   conly = the same members with only the 115/130 view (python, for the averaging check)
#   bash qa/main_v5c_crop/dryrun.sh
set -u
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.worktrees/release_v7
PY=/raid/sroziewski/sandbox/bin/python
cd "$P"
OUT=qa/main_v5c_crop/dryrun
mkdir -p "$OUT"
export PYTHONPATH=$W/src PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/.tmp CUDA_VISIBLE_DEVICES=
# Pipeline B's chains share this machine's CPUs with their dataloaders: cap and nice the inference.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
renice -n 10 $$ > /dev/null
CROP=0.8846153846153846   # repr(115.0 / 130.0), the factor qa/tta2 scored
if [ ! -d export/main_v5_C_crop ]; then
  $PY "$W/scripts/export_kaggle.py" --from-bundle export/main_v5_C --name main_v5_C_crop --export-root export \
    --view-scales 1.0 $CROP --user sroziews --dataset-slug rsna-kad-weights-v5c-crop > "$OUT/derive.log" 2>&1
  echo "derive rc=$?"
fi
for b in main_v5_C main_v5_C_crop; do
  $PY "$W/notebooks/kaggle_inference.py" --data-dir /raid/sroziewski/data/rsna_kad --weights "export/$b" \
    --out "$OUT/submission_$b.csv" --device cpu --fp32 > "$OUT/notebook_$b.log" 2>&1; echo "notebook $b rc=$?"
done
$PY - "$OUT" "$CROP" <<'EOF' 2>&1 | tee "$OUT/checks.txt"
import sys, json, hashlib, re
from pathlib import Path
import numpy as np, pandas as pd
from rsna_kad.constants import ID_COL, LABELS
from rsna_kad.infer import members_from_ensemble, predict_studies, test_study_dirs
out, crop = Path(sys.argv[1]), float(sys.argv[2])
fails = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"); ok or fails.append(name)
rd = lambda p: pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)
ref = rd("export/submission_local_main_v5_C.csv")
base, both = rd(out / "submission_main_v5_C.csv"), rd(out / "submission_main_v5_C_crop.csv")
X = lambda d: d[list(LABELS)].to_numpy(np.float64)
d0 = np.abs(X(base) - X(ref)).max()
check("base-only (release/v7 notebook) == main_v5_C local output (e52fc80)", d0 <= 5e-7, f"max abs {d0:.2e}")
src, der = (json.loads(Path(f"export/{b}/ensemble.json").read_text()) for b in ("main_v5_C", "main_v5_C_crop"))
check("same members/weights/tta/prior/K as main_v5_C",
      [(m["path"], m["weight"], m["tta"]) for m in src["members"]] == [(m["path"], m["weight"], m["tta"]) for m in der["members"]]
      and src["prior"] == der["prior"] and src["K_infer"] == der["K_infer"])
check("view_scales [1.0, 115/130] on every member", all(m["view_scales"] == [1.0, crop] for m in der["members"]))
same = all(hashlib.md5(Path(f"export/main_v5_C/{m['path']}{s}").read_bytes()).digest()
           == hashlib.md5(Path(f"export/main_v5_C_crop/{m['path']}{s}").read_bytes()).digest()
           for m in src["members"] for s in (".safetensors", ".json"))
check("member files byte-identical to main_v5_C", same)
meta = json.loads(Path("export/main_v5_C_crop/dataset-metadata.json").read_text())
check("dataset licence CC-BY-NC-4.0, new slug", meta["licenses"] == [{"name": "CC-BY-NC-4.0"}]
      and meta["id"] == "sroziews/rsna-kad-weights-v5c-crop", json.dumps(meta))
log = Path(out / "notebook_main_v5_C_crop.log").read_text()
check("crop run: 6 members, mode main, 0 prior rows, no traceback",
      "6 members, mode main" in log and re.search(r"\| 0 prior rows", log) is not None and "Traceback" not in log)
# crop-only view through the same code path, for the averaging check
data = Path("/raid/sroziewski/data/rsna_kad")
sample = pd.read_csv(data / "sample_submission.csv", dtype={ID_COL: str})
series = pd.read_csv(data / "test_series.csv", dtype=str)
ms = members_from_ensemble("export/main_v5_C")
for m in ms:
    m.view_scales = (crop,)
po = predict_studies(ms, test_study_dirs(data, series, sample), series, devices=["cpu"], workers=3, fp16=False)
conly = po.set_index(ID_COL).loc[both[ID_COL], list(LABELS)].to_numpy(np.float64)
err = np.abs(X(both) - 0.5 * (X(base) + conly)).max()
check("probe == equal-weight mean of base and crop views", err < 2e-6, f"max abs {err:.2e}")
print(f"INFO  probe vs main_v5_C: mean abs {np.abs(X(both) - X(base)).mean():.4f}, max {np.abs(X(both) - X(base)).max():.4f}")
t = {b: float(re.search(r"inference [0-9.]+s = ([0-9.]+)s/study", Path(out / f"notebook_{b}.log").read_text()).group(1))
     for b in ("main_v5_C", "main_v5_C_crop")}
print(f"INFO  CPU fp32 s/study: base {t['main_v5_C']}, crop {t['main_v5_C_crop']} (x{t['main_v5_C_crop'] / t['main_v5_C']:.2f})")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
EOF
echo DRYRUNDONE
