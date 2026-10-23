#!/usr/bin/env bash
# main_v5_C + maxpool + nnPU probe: 6 main_v5_C members + 3 max-pool + 3 nnPU full-data runs, equal
# weights 1/12, main_v5_C's prior, no TTA, K 24, exported with release/v7, checked on CPU fp32:
#   1. branch equivalence: each max-pool (and one nnPU, one base) member gives the same logits
#      through release/v7's model as through branch maxpool-nnpu's (69a8b8e) on identical inputs
#   2. the notebook loads 12 members, 0 prior rows, over the 3 public studies
#   3. the first 6 members' mean reproduces main_v5_C at the same thread count
#   4. the 12-member output is the mean of the two 6-member halves
#   bash qa/main_v5c_mpnnpu/dryrun.sh
set -u
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.worktrees/release_v7
B=$P/.worktrees/_mpcheck
PY=/raid/sroziewski/sandbox/bin/python
cd "$P"
OUT=qa/main_v5c_mpnnpu/dryrun
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/.tmp CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
renice -n 10 $$ > /dev/null
C=runs/main_armc_convnextv2_t_384_p_full
BASE=$(echo ${C}_s{2026,2126,2226,2326,2426,2526})
MP=$(echo ${C}_maxpool_s{2026,2126,2226})
NN=$(echo ${C}_nnpu_s{2026,2126,2226})

[ -d "$B" ] || git worktree add -q --detach "$B" 69a8b8e
echo "branch check tree: $(git -C "$B" log --oneline -1)"
CHECK="$MP ${C}_nnpu_s2026 ${C}_s2026"
# shellcheck disable=SC2086
PYTHONPATH=$W/src $PY qa/main_v5c_mpnnpu/branch_logits.py "$OUT/logits_v7.npz" $CHECK > "$OUT/logits_v7.log" 2>&1; echo "v7 logits rc=$?"
# shellcheck disable=SC2086
PYTHONPATH=$B/src $PY qa/main_v5c_mpnnpu/branch_logits.py "$OUT/logits_branch.npz" $CHECK > "$OUT/logits_branch.log" 2>&1; echo "branch logits rc=$?"
git worktree remove --force "$B" && echo "branch check tree removed"

if [ ! -d export/main_v5_C_mpnnpu ]; then
  # shellcheck disable=SC2086
  PYTHONPATH=$W/src $PY "$W/scripts/export_kaggle.py" --name main_v5_C_mpnnpu --export-root export --wheels-dir export/wheels \
    --user sroziews --dataset-slug rsna-kad-weights-v5c-mpnnpu --no-tta --k-infer 24 \
    --group "C=${BASE%% *}" ${BASE#* } $MP $NN --prior-of-bundle export/main_v5_C > "$OUT/export.log" 2>&1
  echo "export rc=$?"
fi
PYTHONPATH=$W/src $PY "$W/notebooks/kaggle_inference.py" --data-dir /raid/sroziewski/data/rsna_kad --weights export/main_v5_C_mpnnpu \
  --out "$OUT/submission.csv" --device cpu --fp32 > "$OUT/notebook.log" 2>&1; echo "notebook rc=$?"
PYTHONPATH=$W/src $PY - "$OUT" <<'EOF' 2>&1 | tee "$OUT/checks.txt"
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from rsna_kad.constants import ID_COL, LABELS
from rsna_kad.infer import members_from_ensemble, predict_studies, test_study_dirs
out = Path(sys.argv[1])
fails = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"); ok or fails.append(name)
a, b = np.load(out / "logits_v7.npz"), np.load(out / "logits_branch.npz")
check("logits computed for the same 5 members in both trees", sorted(a.files) == sorted(b.files) and len(a.files) == 5, str(sorted(a.files)))
for k in sorted(a.files):
    d = float(np.abs(a[k] - b[k]).max())
    check(f"release/v7 == maxpool-nnpu branch: {k}", d <= 1e-5, f"max abs logit diff {d:.2e}")
doc, v5c = (json.loads(Path(f"export/{x}/ensemble.json").read_text()) for x in ("main_v5_C_mpnnpu", "main_v5_C"))
names = [m["path"].split("/")[1] for m in doc["members"]]
want = ([f"main_armc_convnextv2_t_384_p_full_s{s}_full" for s in (2026, 2126, 2226, 2326, 2426, 2526)]
        + [f"main_armc_convnextv2_t_384_p_full_maxpool_s{s}_full" for s in (2026, 2126, 2226)]
        + [f"main_armc_convnextv2_t_384_p_full_nnpu_s{s}_full" for s in (2026, 2126, 2226)])
check("12 members: 6 main_v5_C + 3 max-pool + 3 nnPU", names == want)
check("first 6 == main_v5_C's members", names[:6] == [m["path"].split("/")[1] for m in v5c["members"]])
check("weights 1/12, no TTA, no crop view, K 24",
      all(abs(m["weight"] - 1/12) < 1e-12 and m["tta"] is False and "view_scales" not in m for m in doc["members"]) and doc["K_infer"] == 24)
check("prior == main_v5_C's (56687125)", doc["prior"] == v5c["prior"], doc["prior_source"])
mp_flags = [json.loads(Path(f"export/main_v5_C_mpnnpu/models/{n}.json").read_text())["config"]["model"].get("mil_maxpool_residual")
            for n in names]
check("max-pool flag True on the 3 max-pool members only", mp_flags[6:9] == [True] * 3 and not any(mp_flags[:6] + mp_flags[9:]), str(mp_flags))
meta = json.loads(Path("export/main_v5_C_mpnnpu/dataset-metadata.json").read_text())
check("dataset CC-BY-NC-4.0, new slug, no NOTICE", meta["licenses"] == [{"name": "CC-BY-NC-4.0"}]
      and meta["id"] == "sroziews/rsna-kad-weights-v5c-mpnnpu" and not Path("export/main_v5_C_mpnnpu/NOTICE").exists())
log = (out / "notebook.log").read_text()
check("notebook: 12 members, mode main, 12 loaded, 0 prior rows, no traceback",
      "12 members, mode main" in log and len(re.findall(r"INFO loaded ", log)) == 12 and re.search(r"\| 0 prior rows", log) is not None and "Traceback" not in log)
rd = lambda p: pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)
sub, ref = rd(out / "submission.csv"), rd("qa/main_v5c_crop/dryrun/submission_main_v5_C.csv")
data = Path("/raid/sroziewski/data/rsna_kad")
sample = pd.read_csv(data / "sample_submission.csv", dtype={ID_COL: str})
series = pd.read_csv(data / "test_series.csv", dtype=str)
dirs = test_study_dirs(data, series, sample)
ms = members_from_ensemble("export/main_v5_C_mpnnpu")
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
print(f"INFO  probe vs main_v5_C: mean abs {dv.mean():.4f}, max {dv.max():.4f}; new 6 vs main_v5_C 6: mean {np.abs(half[0]-half[1]).mean():.4f}")
t = float(re.search(r"inference [0-9.]+s = ([0-9.]+)s/study", log).group(1))
print(f"INFO  CPU fp32 s/study: {t} vs main_v5_C 105.99 and main_v5_C12 271.88 at the same threads")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
EOF
echo DRYRUNDONE
