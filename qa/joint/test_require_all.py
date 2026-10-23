"""Stub-leg test of build_joint.py --require-all-legs (qa/joint, 2026-10-16).

Builds the joint kernel from tiny stand-in legs (V, P, extra A) that write a random submission for a fake competition
folder, runs it, and checks: all legs ok -> same submission as without the flag; A failing or missing a study -> the
run exits non-zero and leaves no submission.csv (P's file must not survive); flag off + A failing -> the V/P blend.
    python qa/joint/test_require_all.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = ["a", "b", "c"]
IDS = [f"1.2.{i}" for i in range(20)]

LEG = """import os, sys, time
import numpy as np, pandas as pd
T0 = time.time()
ids = [f"1.2.{{i}}" for i in range(20)]
rng = np.random.default_rng({seed})
df = pd.DataFrame(rng.random((len(ids), 3)), columns={labels!r})
df.insert(0, "StudyInstanceUID", ids)
mode = "{mode}"
if mode == "fail":
    sys.exit(1)
if mode == "short":
    df = df.iloc[1:]
out = os.environ.get("{env}") or os.getcwd()
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, "submission.csv"), index=False)
"""


def leg(d, name, seed, mode="ok", env="NO_SUCH_ENV"):
    p = d / f"{name}_{mode}.py"
    p.write_text(LEG.format(seed=seed, labels=LABELS, mode=mode, env=env))
    return p


def run(d, case, a_mode, require, v_mode="ok", w_partner=None, a_w=0.2):
    fake = d / case
    comp = fake / "input/competitions/rsna-knee-abnormality-detection"
    comp.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": IDS, **{k: 0.5 for k in LABELS}}).to_csv(comp / "sample_submission.csv",
                                                                             index=False)
    meta = d / "meta.json"
    meta.write_text(json.dumps({"dataset_sources": [], "docker_image": "x"}))
    out = fake / "kernel"
    cmd = [sys.executable, str(HERE / "build_joint.py"), "--fake", str(fake), "--inner", str(leg(d, "V", 1, v_mode)),
           "--inner-meta", str(meta), "--partner-src", str(leg(d, "P", 2)),
           "--extra", f"A:{a_w}:{leg(d, 'A', 3, a_mode, 'HX48_WORK')}:HX48_WORK:hx48", "--out", str(out)]
    if require:
        cmd.append("--require-all-legs")
    if w_partner is not None:
        cmd += ["--w-partner", str(w_partner)]
    subprocess.run(cmd, check=True, capture_output=True)
    work = fake / "working"
    work.mkdir()
    r = subprocess.run([sys.executable, str(out / "joint.py")], capture_output=True, text=True,
                       env=dict(os.environ, BLEND_WORK=str(work)))
    sub = work / "submission.csv"
    r.report = json.loads((work / "joint_report.json").read_text())
    return r.returncode, (pd.read_csv(sub, dtype=str).set_index("StudyInstanceUID").astype(float)
                          if sub.exists() else None), r


def main():
    fails = []
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        rc0, base, r0 = run(d, "ok_off", "ok", False)
        rc1, req, r1 = run(d, "ok_on", "ok", True)
        rc2, sub2, r2 = run(d, "fail_on", "fail", True)
        rc3, sub3, r3 = run(d, "short_on", "short", True)
        rc4, sub4, r4 = run(d, "fail_off", "fail", False)
        # V would fail if it ran; at weight 0 it must not run at all
        rc5, sub5, r5 = run(d, "pa_on", "ok", True, v_mode="fail", w_partner=1.0, a_w=0.4)
        pr = pd.read_csv(d / "pa_on/tmp/joint/P.csv", dtype=str).set_index("StudyInstanceUID").astype(float).rank(pct=True)
        ar = pd.read_csv(d / "pa_on/tmp/joint/A.csv", dtype=str).set_index("StudyInstanceUID").astype(float).rank(pct=True)
        checks = {
            "ok, flag off: exit 0, blend of V/P/A": rc0 == 0 and base is not None
            and r0.report["weights_used"] == {"V": 0.4, "P": 0.4, "A": 0.2},
            "ok, flag on: exit 0, identical submission": rc1 == 0 and req is not None and np.array_equal(base, req),
            "A fails, flag on: exit != 0, no submission.csv": rc2 != 0 and sub2 is None
            and r2.report["submission"] is None,
            "A fails, flag on: error names leg A": "legs dropped: ['A']" in r2.stderr,
            "A misses a study, flag on: exit != 0, no submission.csv": rc3 != 0 and sub3 is None
            and "missed 1 studies" in r3.report["A"]["error"],
            "A fails, flag off: exit 0 with the V/P blend": rc4 == 0 and sub4 is not None
            and r4.report["weights_used"] == {"V": 0.5, "P": 0.5} and not np.array_equal(sub4, base),
            "w_partner 1.0, flag on: V not run, exit 0, .6 P + .4 A": rc5 == 0 and "V" not in r5.report
            and r5.report["weights_used"] == {"P": 0.6, "A": 0.4}
            and np.allclose(sub5.to_numpy(), (0.6 * pr + 0.4 * ar).loc[sub5.index, sub5.columns].to_numpy()),
        }
    for k, ok in checks.items():
        print("PASS" if ok else "FAIL", k)
        if not ok:
            fails.append(k)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
