#!/usr/bin/env bash
# Probe "main_v5_C + silentq synovitis" (user-approved, exactly one submission), run unattended on the server:
#   11 labels = main_v5_C (6 Cfull s2026-2526, as 56687125); synovitis = mean of the 3 silentq_syn full-data
#   models. Groups A (6) / S (3), --group-weights A=1,S=0, --label-group-weights synovitis:A=0,S=1,
#   main_v5_C's prior, no TTA, K 24, CC-BY-NC-4.0, new private dataset + kernel.
# Waits for qa/silentq_syn/DONE, then: guard checks -> export -> CPU dry run -> dataset -> kernel -> kernel
# checks -> submit -> score. Every gate writes one line to $ST; the first failing gate writes STOPPED and
# exits WITHOUT submitting. Nothing is submitted twice: a SUBMITTED line in $ST makes a rerun stop.
#   bash scripts/run_screen.sh silentq bash qa/main_v5c_silentq/pipeline.sh
set -u
P=/raid/sroziewski/sandbox/rsna_kad
W=$P/.worktrees/release_v7
PY=/raid/sroziewski/sandbox/bin/python
cd "$P"
Q=qa/main_v5c_silentq
OUT=$Q/dryrun
ST=$Q/STATUS
NAME=main_v5_C_silentq
SLUG=rsna-kad-weights-v5c-silentq
KSLUG=rsna-kad-main-v5c-silentq
MSG="main_v5_C + silentq synovitis (probe)"
mkdir -p "$OUT" "$Q/kernel"
say() { echo "$(date -Is) $*" | tee -a "$ST"; }
stop() { say "STOPPED: $*"; exit 1; }
grep -q "^.* SUBMITTED" "$ST" 2>/dev/null && stop "a submission is already recorded in $ST - refusing to submit twice"
export PYTHONPATH=$W/src PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/.tmp CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export KAGGLE_CONFIG_DIR=$P/.kaggle KERNEL_META=$P/$Q/kernel/kernel-metadata.json
C=$W/scripts/kaggle_cli.sh
renice -n 10 $$ > /dev/null

# 1. wait for the models (poll on the server; up to 30 h)
say "waiting for qa/silentq_syn/DONE"
for i in $(seq 1 360); do [ -f qa/silentq_syn/DONE ] && break; sleep 300; done
[ -f qa/silentq_syn/DONE ] || stop "qa/silentq_syn/DONE never appeared"
say "DONE present"
A=$(echo runs/main_armc_convnextv2_t_384_p_full_s{2026,2126,2226,2326,2426,2526})
S=$(echo runs/main_armc_convnextv2_t_384_p_full_silentq_syn_s{2026,2126,2226})

# 2. the 3 new runs: complete, and the same inference architecture as the base members
$PY - $S <<'EOF' >> "$ST" 2>&1 || stop "S runs incomplete or architecture differs (see above)"
import json, sys
from pathlib import Path
base = json.loads(Path("runs/main_armc_convnextv2_t_384_p_full_s2026/full/bundle.json").read_text())
ok = True
for rd in sys.argv[1:]:
    f = Path(rd) / "full"
    m = json.loads((f / "bundle.json").read_text())
    done = (f / "metrics.json").exists() and not (f / "last.pt").exists() and m["metrics"].get("epoch") == m["config"]["epochs"] - 1
    same = all(m["arch"][k] == base["arch"][k] for k in ("backbone", "img_px", "slots", "view_crop_mm", "K_infer"))
    same = same and m["config"]["model"].get("head") == base["config"]["model"].get("head")
    print(f"  {rd}: complete={done} same_arch={same} targets={m['config'].get('targets_version')}")
    ok &= done and same
sys.exit(0 if ok else 1)
EOF
say "S runs complete, architecture = base arm C"

# 3. export
if [ ! -d "export/$NAME" ]; then
  # shellcheck disable=SC2086
  $PY "$W/scripts/export_kaggle.py" --name "$NAME" --export-root export --wheels-dir export/wheels \
    --user sroziews --dataset-slug "$SLUG" --no-tta --k-infer 24 \
    --group "A=${A%% *}" ${A#* } --group "S=${S%% *}" ${S#* } --group-weights A=1,S=0 \
    --label-group-weights synovitis:A=0,S=1 --prior-of-bundle export/main_v5_C > "$OUT/export.log" 2>&1 \
    || stop "export refused: $(tail -3 "$OUT/export.log" | tr '\n' ' ')"
fi
say "exported export/$NAME"

# 4. CPU dry run over the 3 public studies + checks
$PY "$W/notebooks/kaggle_inference.py" --data-dir /raid/sroziewski/data/rsna_kad --weights "export/$NAME" \
  --out "$OUT/submission.csv" --device cpu --fp32 > "$OUT/notebook.log" 2>&1 || stop "notebook failed on CPU"
$PY - "$OUT" "$NAME" <<'EOF' > "$OUT/checks.txt" 2>&1
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
from rsna_kad.constants import ID_COL, LABELS, LABEL_KEYS
from rsna_kad.infer import members_from_ensemble, predict_studies, test_study_dirs
out, name = Path(sys.argv[1]), sys.argv[2]
bundle = Path("export") / name
J = LABEL_KEYS.index("synovitis"); oj = [j for j in range(12) if j != J]
fails = []
def check(n, ok, d=""):
    print(f"{'PASS' if ok else 'FAIL'}  {n}  {d}"); ok or fails.append(n)
doc, v5c = json.loads((bundle / "ensemble.json").read_text()), json.loads(Path("export/main_v5_C/ensemble.json").read_text())
a = [m for m in doc["members"] if m["group"] == "A"]; s = [m for m in doc["members"] if m["group"] == "S"]
check("6 A (main_v5_C's) + 3 S", [m["path"] for m in a] == [m["path"] for m in v5c["members"]] and len(s) == 3 and all("silentq_syn" in m["path"] for m in s))
check("A: 1/6 on 11 labels, 0 on synovitis", all(np.allclose([m["label_weights"][j] for j in oj], 1/6) and m["label_weights"][J] == 0.0 for m in a))
check("S: 1/3 on synovitis, 0 elsewhere", all(np.isclose(m["label_weights"][J], 1/3) and all(m["label_weights"][j] == 0.0 for j in oj) for m in s))
check("prior == main_v5_C's", doc["prior"] == v5c["prior"], doc["prior_source"])
check("no TTA, K 24", all(m["tta"] is False for m in doc["members"]) and doc["K_infer"] == 24)
meta = json.loads((bundle / "dataset-metadata.json").read_text())
check("licence CC-BY-NC-4.0, no NOTICE", doc["license"] == "CC-BY-NC-4.0" and meta["licenses"] == [{"name": "CC-BY-NC-4.0"}] and not (bundle / "NOTICE").exists())
log = (out / "notebook.log").read_text()
check("notebook: 9 members, mode main, 9 loaded, 0 prior rows", "9 members, mode main" in log and len(re.findall(r"INFO loaded ", log)) == 9
      and re.search(r"\| 0 prior rows", log) is not None and "Traceback" not in log)
rd = lambda p: pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)
sub, ref = rd(out / "submission.csv"), rd("qa/main_v5c_crop/dryrun/submission_main_v5_C.csv")
X = lambda d: d[list(LABELS)].to_numpy(np.float64)
check("11 labels == main_v5_C exactly (same threads)", np.abs(X(sub)[:, oj] - X(ref)[:, oj]).max() == 0.0,
      f"max abs {np.abs(X(sub)[:, oj] - X(ref)[:, oj]).max():.2e}")
data = Path("/raid/sroziewski/data/rsna_kad")
sample = pd.read_csv(data / "sample_submission.csv", dtype={ID_COL: str}); series = pd.read_csv(data / "test_series.csv", dtype=str)
dirs = test_study_dirs(data, series, sample); ids = list(sub[ID_COL])
ms = members_from_ensemble(bundle)
ss = [m for m in ms if "silentq_syn" in m.prefix]
for m in ss:
    m.label_weights, m.weight = None, 1.0
ps = predict_studies(ss, dirs, series, devices=["cpu"], workers=3, fp16=False).set_index(ID_COL).loc[ids, list(LABELS)].to_numpy(np.float64)
full = predict_studies(members_from_ensemble(bundle), dirs, series, devices=["cpu"], workers=3, fp16=False).set_index(ID_COL).loc[ids, list(LABELS)].to_numpy(np.float64)
check("full precision: synovitis == mean of the 3 S members", abs(full[:, J] - ps[:, J]).max() < 1e-12, f"max abs {abs(full[:, J] - ps[:, J]).max():.2e}")
check("notebook CSV == the ensemble at full precision (float32 + 6 decimals)", np.abs(X(sub) - full).max() <= 5.2e-7, f"max abs {np.abs(X(sub) - full).max():.2e}")
np.save(out / "local_s_only.npy", ps)
print(f"INFO  synovitis moved vs main_v5_C: mean abs {np.abs(X(sub)[:, J] - X(ref)[:, J]).mean():.4f} max {np.abs(X(sub)[:, J] - X(ref)[:, J]).max():.4f}")
t = float(re.search(r"inference [0-9.]+s = ([0-9.]+)s/study", log).group(1))
print(f"INFO  CPU fp32 s/study {t} vs main_v5_C 105.99 (x{t / 105.99:.2f}); main_v5_C hidden run ~1.2 h -> ~{1.2 * t / 105.99:.1f} h")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
EOF
cat "$OUT/checks.txt" >> "$ST"
grep -q "^RESULT PASS" "$OUT/checks.txt" || stop "dry-run checks failed"
say "dry run PASS"

# 5. dataset: create, then the FILE LISTING must show all 9 members + VERSION, and the licence must read back
if ! grep -q "dataset created" "$ST"; then
  bash "$C" dataset-create "$P/export/$NAME" > "$OUT/push.log" 2>&1 || stop "dataset create failed: $(tail -2 "$OUT/push.log" | tr '\n' ' ')"
  say "dataset created"
fi
VB=$(wc -c < "export/$NAME/VERSION")
for i in $(seq 1 30); do
  L=$(bash "$C" raw datasets files "sroziews/$SLUG" --page-size 200 2>&1)
  n=$(echo "$L" | grep -c "\.safetensors"); v=$(echo "$L" | grep -E "^VERSION" | awk '{print $2}')
  [ "$n" = 9 ] && [ "$v" = "$VB" ] && break; sleep 30
done
want=$(ls "export/$NAME/models/"*.safetensors | xargs -n1 basename | sort | md5sum | cut -c1-8)
got=$(echo "$L" | grep "\.safetensors" | awk '{print $1}' | xargs -n1 basename | sort | md5sum | cut -c1-8)
[ "$n" = 9 ] && [ "$v" = "$VB" ] && [ "$want" = "$got" ] || stop "dataset listing incomplete: $n/9 members, VERSION $v (want $VB), names match=$([ "$want" = "$got" ] && echo yes || echo no)"
rm -rf .tmp/meta_silentq; bash "$C" raw datasets metadata "sroziews/$SLUG" -p .tmp/meta_silentq > /dev/null 2>&1
grep -q "CC BY-NC 4.0" .tmp/meta_silentq/*.json && grep -q '"isPrivate": true' .tmp/meta_silentq/*.json || stop "dataset licence/privacy did not read back as CC BY-NC 4.0 + private"
say "dataset listing complete (9 members, VERSION $VB B), private, CC BY-NC 4.0"

# 6. kernel: push, wait, verify
cp "$W/notebooks/kaggle_inference.py" "$Q/kernel/kaggle_inference.py"
printf '{\n "id": "sroziews/%s",\n "title": "%s",\n "code_file": "kaggle_inference.py",\n "language": "python",\n "kernel_type": "script",\n "is_private": true,\n "enable_gpu": true,\n "enable_tpu": false,\n "enable_internet": false,\n "dataset_sources": ["sroziews/%s"],\n "competition_sources": ["rsna-knee-abnormality-detection"],\n "kernel_sources": [],\n "model_sources": []\n}\n' \
  "$KSLUG" "$KSLUG" "$SLUG" > "$Q/kernel/kernel-metadata.json"
say "kernel notebook md5 $(md5sum < "$Q/kernel/kaggle_inference.py" | cut -c1-32) ($(git -C "$W" log --oneline -1))"
bash "$C" kernel-push > "$OUT/kernel_push.log" 2>&1 || stop "kernel push failed"
KV=$(grep -oE "Kernel version [0-9]+" "$OUT/kernel_push.log" | awk '{print $3}')
[ -n "$KV" ] || stop "no kernel version in push output: $(cat "$OUT/kernel_push.log")"
for i in $(seq 1 60); do s=$(bash "$C" kernel-status 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z]+"); case "$s" in *RUNNING*|*QUEUED*|"") sleep 30;; *) break;; esac; done
[ "$s" = "KernelWorkerStatus.COMPLETE" ] || stop "kernel v$KV ended $s"
rm -rf "$Q/kernel_out"; bash "$C" kernel-output "$Q/kernel_out" > /dev/null 2>&1
$PY - "$Q" <<'EOF' > "$Q/kernel_checks.txt" 2>&1
import sys, re, glob
from pathlib import Path
import numpy as np, pandas as pd
from rsna_kad.constants import ID_COL, LABELS, LABEL_KEYS
q = Path(sys.argv[1]); J = LABEL_KEYS.index("synovitis"); oj = [j for j in range(12) if j != J]
log = Path(glob.glob(str(q / "kernel_out" / "*.log"))[0]).read_text()
fails = []
def check(n, ok, d=""):
    print(f"{'PASS' if ok else 'FAIL'}  {n}  {d}"); ok or fails.append(n)
check("kernel log: 9 members, mode main, dataset v5c-silentq", re.search(r"rsna-kad-weights-v5c-silentq: 9 members, mode main", log) is not None)
check("kernel log: 9 loaded, 0 prior rows, no traceback", len(re.findall(r"INFO loaded ", log)) == 9 and re.search(r"\| 0 prior rows", log) is not None and "Traceback" not in log)
m = re.search(r"inference [0-9.]+s = [0-9.]+s/study \(prep [0-9.]+s, gpu [0-9.]+s per study\)", log)
print("INFO ", m.group(0) if m else "no timing line", "|", ", ".join(sorted(set(re.findall(r"torch [0-9.+a-z]+", log)))))
rd = lambda p: pd.read_csv(p).sort_values(ID_COL).reset_index(drop=True)
k, l, v = rd(q / "kernel_out" / "submission.csv"), rd(q / "dryrun" / "submission.csv"), rd("qa/main_v5/kernel_out_v5c/submission.csv")
check("same ids and columns", list(k[ID_COL]) == list(l[ID_COL]) == list(v[ID_COL]) and list(k.columns) == [ID_COL, *LABELS])
K_, L_, V_ = (x[list(LABELS)].to_numpy(np.float64) for x in (k, l, v))
d = np.abs(K_ - L_)
check("kernel-vs-local <= 1e-3", d.max() <= 1e-3 and not np.isnan(K_).any(), f"max {d.max():.2e} mean {d.mean():.2e} (synovitis {d[:, J].max():.2e})")
dA = np.abs(K_[:, oj] - V_[:, oj]).max()
check("11 labels reproduce main_v5_C's kernel output", dA <= 1e-3, f"max {dA:.2e}")
dk, dl = np.abs(K_[:, J] - V_[:, J]), np.abs(L_[:, J] - rd("qa/main_v5c_crop/dryrun/submission_main_v5_C.csv")[list(LABELS)].to_numpy(np.float64)[:, J])
print(f"INFO  synovitis shift vs main_v5_C: kernel mean {dk.mean():.4f} max {dk.max():.4f} | local mean {dl.mean():.4f} max {dl.max():.4f}")
print("RESULT", "PASS" if not fails else f"FAIL {fails}")
EOF
cat "$Q/kernel_checks.txt" >> "$ST"
grep -q "^RESULT PASS" "$Q/kernel_checks.txt" || stop "kernel checks failed"
say "kernel v$KV verified"

# 7. submit exactly once (Kaggle counts days in UTC; wait for a free slot rather than exceed 5)
for i in $(seq 1 30); do
  used=$(bash "$C" submissions 2>/dev/null | awk -v d="$(date -u +%F)" '$3==d' | wc -l)
  [ "$used" -lt 5 ] && break; say "5 submissions already on $(date -u +%F) UTC - waiting"; sleep 1800
done
bash "$C" submit "$KV" "$MSG" > "$OUT/submit.log" 2>&1 || stop "submit failed: $(cat "$OUT/submit.log")"
say "SUBMITTED kernel v$KV: $MSG ($(tail -1 "$OUT/submit.log"))"
sleep 10
SID=$(bash "$C" submissions 2>/dev/null | grep -F "silentq synovitis" | awk 'NR==1 {print $1}')
[ -n "$SID" ] || stop "submission id not found in the list - check by hand, do NOT resubmit"
say "submission id $SID"

# 8. score
for i in $(seq 1 96); do
  r=$(bash "$C" submissions 2>/dev/null | grep -E "^$SID " | awk '{print $(NF-1), $NF}')
  case "$r" in *PENDING*|"") sleep 300;; *) break;; esac
done
say "SCORE $SID: $r"
echo PIPELINEDONE
