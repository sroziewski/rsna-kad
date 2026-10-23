"""Stub-leg test of build_joint.py --diag-encode (qa/joint, 2026-10-16).

Legs V, P and extra A are tiny scripts writing random predictions for 20 fake studies with the 12 real finding
columns. Each case forces one outcome on the real code path (A exits 1, A writes no file, A misses a study, A sleeps
past its timeout, too little time left to start A, P fails, V fails) and checks that the submission blanks exactly the
first <code> DIAG_COLS (constant 0.5) and leaves the other columns equal to the fallback blend. diag_code's timeout
budget buckets are checked on synthetic reports.
    python qa/joint/test_diag_encode.py
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
DIAG_COLS = ["ACL", "Medial OA", "Effusion", "MCL", "Medial Meniscus", "Baker's", "Fracture", "Contusion"]
IDS = [f"1.2.{i}" for i in range(20)]

LEG = """import os, sys, time
import numpy as np, pandas as pd
T0 = time.time()
ids = [f"1.2.{{i}}" for i in range(20)]
df = pd.DataFrame(np.random.default_rng({seed}).random((len(ids), 12)), columns={labels!r})
df.insert(0, "StudyInstanceUID", ids)
mode = "{mode}"
if mode == "fail":
    sys.exit(1)
if mode == "nofile":
    sys.exit(0)
if mode == "sleep":
    time.sleep(30)
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


def build(d, case, v="ok", p="ok", a="ok", diag=True):
    fake = d / case
    comp = fake / "input/competitions/rsna-knee-abnormality-detection"
    comp.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": IDS, **{k: 0.5 for k in LABELS}}).to_csv(comp / "sample_submission.csv",
                                                                             index=False)
    meta = d / "meta.json"
    meta.write_text(json.dumps({"dataset_sources": [], "docker_image": "x"}))
    out = fake / "kernel"
    # 0.002 h = 7.2 s: long enough for a normal stub leg, short enough for the sleeping one to time out
    subprocess.run([sys.executable, str(HERE / "build_joint.py"), "--fake", str(fake), "--inner", str(leg(d, "V", 1, v)),
                    "--inner-meta", str(meta), "--partner-src", str(leg(d, "P", 2, p)), "--extra-max-h", "0.002",
                    "--extra", f"A:0.2:{leg(d, 'A', 3, a, 'HX48_WORK')}:HX48_WORK:hx48", "--out", str(out)]
                   + (["--diag-encode"] if diag else []), check=True, capture_output=True)
    work = fake / "working"
    work.mkdir()
    return out / "joint.py", work


def read(work):
    sub = pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    return sub, json.loads((work / "joint_report.json").read_text())


def run(d, case, diag=True, **modes):
    joint, work = build(d, case, diag=diag, **modes)
    r = subprocess.run([sys.executable, str(joint)], capture_output=True, text=True,
                       env=dict(os.environ, BLEND_WORK=str(work)))
    if r.returncode:
        raise RuntimeError(f"{case}: exit {r.returncode}\n{r.stderr[-800:]}")
    return read(work)


def load_module(joint, work):
    os.environ["BLEND_WORK"] = str(work)
    spec = importlib.util.spec_from_file_location(f"joint_{work.parent.name}", joint)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blanked_ok(sub, ref, code):
    """first <code> DIAG_COLS constant 0.5, the rest identical to the reference blend"""
    blank, keep = DIAG_COLS[:code], [c for c in LABELS if c not in DIAG_COLS[:code]]
    return bool((sub[blank] == 0.5).all().all()) and np.allclose(sub[keep].to_numpy(), ref[keep].to_numpy())


def main():
    checks = {}
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        ok, rep_ok = run(d, "ok")
        ok_plain, _ = run(d, "ok_plain", diag=False)
        vp, _ = run(d, "vp_ref", a="fail", diag=False)  # the plain V/P fallback blend, nothing blanked
        checks["all legs ok -> code 0, submission == build without the flag"] = (
            rep_ok["diag_code"] == 0 and np.allclose(ok.to_numpy(), ok_plain.to_numpy())
            and rep_ok["weights_used"] == {"V": 0.4, "P": 0.4, "A": 0.2})
        for case, mode, code, reason in (("fail", "fail", 5, "failed"), ("nofile", "nofile", 5, "failed"),
                                         ("short", "short", 6, "rejected"), ("sleep", "sleep", 3, "timeout")):
            sub, rep = run(d, case, a=mode)
            checks[f"A {mode} -> code {code} ({reason}), V/P blend kept"] = (
                rep["diag_code"] == code and rep["A"]["reason"] == reason and blanked_ok(sub, vp, code)
                and rep["weights_used"] == {"V": 0.5, "P": 0.5})
        sub, rep = run(d, "pfail", p="fail")
        checks["P fails -> code 7"] = rep["diag_code"] == 7 and set(rep["weights_used"]) == {"V", "A"}
        sub, rep = run(d, "vfail", v="fail")
        checks["V fails -> code 8"] = rep["diag_code"] == 8 and set(rep["weights_used"]) == {"P", "A"}
        # too little time left before A: run main() in-process with a shrunken budget
        joint, work = build(d, "skip")
        mod = load_module(joint, work)
        orig_left, calls = mod.left, {"n": 0}

        def left_after_vp():
            # calls 1-3 are V's timeout, P's start check and P's timeout; the 4th, A's start check, sees 100 s
            calls["n"] += 1
            return orig_left() if calls["n"] <= 3 else 100.0
        mod.left = left_after_vp
        mod.main()
        sub, rep = read(work)
        checks["too little time before A -> code 4 (skipped)"] = (rep["diag_code"] == 4 and rep["A"]["reason"] == "skipped"
                                                                  and blanked_ok(sub, vp, 4))
        # timeout budget buckets
        buckets = {3.0 * 3600: 1, 2.9 * 3600: 1, 2.0 * 3600: 2, 1.5 * 3600: 2, 1.0 * 3600: 3, 0: 3}
        checks["timeout budget buckets 1/2/3"] = all(
            mod.diag_code({"A": {"error": "x", "reason": "timeout", "budget_s": b}}) == c for b, c in buckets.items())
        checks["V/P failures outrank A"] = (mod.diag_code({"V": {"error": "x"}, "A": {"error": "x", "reason": "failed"}}) == 8
                                           and mod.diag_code({"P": {"error": "x"}, "A": {"error": "x", "reason": "failed"}}) == 7)
    fails = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print("PASS" if v else "FAIL", k)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
