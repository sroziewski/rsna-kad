#!/usr/bin/env bash
# main_v5_C + 0.2 B_r0 on the weak labels: A = main_v5_C's 6 Cfull members, B = the 10 pipeline-B round-0
# fold models (DINOv3 ViT-B/16 @288). Labels synovitis, lat_oa, pf_oa, lat_men = 0.8 A + 0.2 B; the other
# 8 = A. main_v5_C's prior, no TTA, A at K 24, B at its own bundle settings. Exported with release/v7,
# checked on CPU fp32 over the 3 public studies (8 threads, like the main_v5_C reference run):
#   bash qa/main_v5c_bweak/dryrun.sh
set -u
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.worktrees/release_v7
PY=/raid/sroziewski/sandbox/bin/python
cd "$P"
OUT=qa/main_v5c_bweak/dryrun
mkdir -p "$OUT"
export PYTHONPATH=$W/src PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/.tmp CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
renice -n 10 $$ > /dev/null
A=$(echo runs/main_armc_convnextv2_t_384_p_full_s{2026,2126,2226,2326,2426,2526})
B=$(echo .worktrees/pipeline_b/runs/b_vitb_dinov3_288_r0_s{2026,2126})
if [ ! -d export/main_v5_C_bweak ]; then
  # shellcheck disable=SC2086
  $PY "$W/scripts/export_kaggle.py" --name main_v5_C_bweak --export-root export --wheels-dir export/wheels \
    --user sroziews --dataset-slug rsna-kad-weights-v5c-bweak --no-tta \
    --group "A=${A%% *}" ${A#* } --group "B=${B%% *}" ${B#* } --group-weights A=1,B=0 \
    --label-group-weights synovitis,lat_oa,pf_oa,lat_men:A=0.8,B=0.2 \
    --prior-of-bundle export/main_v5_C > "$OUT/export.log" 2>&1
  echo "export rc=$?"
fi
[ -f export/main_v5_C_bweak/ensemble.json ] || { echo "no export"; exit 1; }
$PY "$W/notebooks/kaggle_inference.py" --data-dir /raid/sroziewski/data/rsna_kad --weights export/main_v5_C_bweak \
  --out "$OUT/submission.csv" --device cpu --fp32 > "$OUT/notebook.log" 2>&1; echo "notebook rc=$?"
$PY - "$OUT" <<'EOF' 2>&1 | tee "$OUT/checks.txt"
import sys, json, re, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from rsna_kad.constants import ID_COL, LABELS, LABEL_KEYS
from rsna_kad.infer import members_from_ensemble, predict_studies, test_study_dirs
out = Path(sys.argv[1]); bundle = Path("export/main_v5_C_bweak")
WEAK = ["synovitis", "lat_oa", "pf_oa", "lat_men"]
wj = [LABEL_KEYS.index(k) for k in WEAK]; oj = [j for j in range(12) if j not in wj]
fails = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"); ok or fails.append(name)
doc, v5c = json.loads((bundle / "ensemble.json").read_text()), json.loads(Path("export/main_v5_C/ensemble.json").read_text())
a = [m for m in doc["members"] if m["group"] == "A"]; b = [m for m in doc["members"] if m["group"] == "B"]
check("6 A (main_v5_C's) + 10 B members", [m["path"] for m in a] == [m["path"] for m in v5c["members"]] and len(b) == 10)
check("A label weights: 1/6 on 8 labels, 0.8/6 on the 4 weak", all(np.allclose([m["label_weights"][j] for j in oj], 1/6) and np.allclose([m["label_weights"][j] for j in wj], 0.8/6) for m in a))
check("B label weights: 0 on 8 labels, 0.02 on the 4 weak", all(all(m["label_weights"][j] == 0.0 for j in oj) and np.allclose([m["label_weights"][j] for j in wj], 0.02) for m in b))
check("label_groups", all(doc["label_groups"][k] == ({"A": 0.8, "B": 0.2} if k in WEAK else {"A": 1.0, "B": 0.0}) for k in LABEL_KEYS))
check("prior == main_v5_C's", doc["prior"] == v5c["prior"], doc["prior_source"])
check("no TTA; A K24 @384, B K16 @288", all(m["tta"] is False for m in doc["members"])
      and all(json.loads((bundle / (m["path"] + ".json")).read_text())["arch"]["K_infer"] == 24 and m["img_px"] == 384 for m in a)
      and all(json.loads((bundle / (m["path"] + ".json")).read_text())["arch"]["K_infer"] == 16 and m["img_px"] == 288 for m in b))
check("licence other + verbatim LICENSE-DINOv3.md", doc["license"] == "other"
      and hashlib.md5((bundle / "LICENSE-DINOv3.md").read_bytes()).hexdigest() == "712846e16e4d72d12ae1052fe7e00133")
notice = (bundle / "NOTICE").read_text()
check("NOTICE names the 10 B members and no A member", all(m["path"] in notice for m in b) and not any(m["path"] in notice for m in a))
meta = json.loads((bundle / "dataset-metadata.json").read_text())
check("dataset-metadata: other, new slug, description", meta["licenses"] == [{"name": "other"}] and meta["id"] == "sroziews/rsna-kad-weights-v5c-bweak" and "NOTICE" in meta.get("description", ""))
log = (out / "notebook.log").read_text()
check("notebook: 16 members, mode main, 16 loaded, 0 prior rows, no traceback",
      "16 members, mode main" in log and len(re.findall(r"INFO loaded ", log)) == 16 and re.search(r"\| 0 prior rows", log) is not None and "Traceback" not in log)
rd = lambda p: pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)
sub, ref = rd(out / "submission.csv"), rd("qa/main_v5c_crop/dryrun/submission_main_v5_C.csv")
X = lambda d: d[list(LABELS)].to_numpy(np.float64)
d8 = np.abs(X(sub)[:, oj] - X(ref)[:, oj]).max()
check("8 untouched labels == main_v5_C exactly (same threads)", d8 == 0.0, f"max abs {d8:.2e}")
data = Path("/raid/sroziewski/data/rsna_kad")
sample = pd.read_csv(data / "sample_submission.csv", dtype={ID_COL: str})
series = pd.read_csv(data / "test_series.csv", dtype=str)
dirs = test_study_dirs(data, series, sample); ids = list(sub[ID_COL])
ms = members_from_ensemble(bundle)
def run(members):
    p = predict_studies(members, dirs, series, devices=["cpu"], workers=3, fp16=False)
    return p.set_index(ID_COL).loc[ids, list(LABELS)].to_numpy(np.float64)
ma = [m for m in ms if "b_vitb" not in m.prefix]; mb = [m for m in ms if "b_vitb" in m.prefix]
for m in ma + mb:
    m.label_weights = None
    m.weight = 1.0
pa, pb = run(ma), run(mb)
ms = members_from_ensemble(bundle)
full = run(ms)
want = pa.copy(); want[:, wj] = 0.8 * pa[:, wj] + 0.2 * pb[:, wj]
dfp = np.abs(full - want).max()
check("full precision: ensemble == A on 8 labels, 0.8 A + 0.2 B on 4", dfp < 1e-12, f"max abs {dfp:.2e}")
dcsv = np.abs(X(sub) - full).max()
check("notebook CSV == the same ensemble at full precision (CSV float32 + 6 decimals)", dcsv <= 5.2e-7, f"max abs {dcsv:.2e}")
np.savez(out / "local_parts.npz", ids=np.array(ids), a=pa, b=pb, full=full)
print(f"INFO  weak labels moved vs main_v5_C: mean abs {np.abs(X(sub)[:, wj] - X(ref)[:, wj]).mean():.4f}, max {np.abs(X(sub)[:, wj] - X(ref)[:, wj]).max():.4f}")
t = float(re.search(r"inference [0-9.]+s = ([0-9.]+)s/study", log).group(1))
print(f"INFO  CPU fp32 s/study: {t} (main_v5_C 105.99 at the same threads)")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
EOF
echo DRYRUNDONE
