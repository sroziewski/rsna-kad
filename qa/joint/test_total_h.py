"""Stub-leg test of build_joint.py --total-h (qa/joint, 2026-10-20).

Rendering: the default build is identical to an explicit --total-h 8.6 and renders `TOTAL_S = 8.6 * 3600`; --total-h
8.8 changes that one line and nothing else; values outside 1-8.85 h stop the build before any joint.py is written.
Behaviour: the legs are the stub scripts of test_diag_encode.py, main() runs in-process, and once P returns the wrapper
clock is fast-forwarded (T0 moved back) to the same elapsed time in both builds, so only the rendered TOTAL_S differs.
The minimum --total-h of 1 h rules out a real budget short enough to exhaust in a test.
    python qa/joint/test_total_h.py
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from test_diag_encode import IDS, LABELS, LEG, load_module, read

HERE = Path(__file__).resolve().parent
A_SLEEP_S = 8
SLEEP_LEG = LEG.replace("time.sleep(30)", f"time.sleep({A_SLEEP_S})")
assert SLEEP_LEG != LEG


def leg(d, name, seed, mode="ok", env="NO_SUCH_ENV"):
    p = d / f"{name}_{mode}.py"
    p.write_text(SLEEP_LEG.format(seed=seed, labels=LABELS, mode=mode, env=env))
    return p


def fake_root(d, case):
    fake = d / case
    comp = fake / "input/competitions/rsna-knee-abnormality-detection"
    comp.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": IDS, **{k: 0.5 for k in LABELS}}).to_csv(comp / "sample_submission.csv",
                                                                             index=False)
    (d / "meta.json").write_text(json.dumps({"dataset_sources": [], "docker_image": "x"}))
    return fake


def build_cmd(d, fake, out, total_h=None, a="sleep"):
    cmd = [sys.executable, str(HERE / "build_joint.py"), "--fake", str(fake), "--inner", str(leg(d, "V", 1)),
           "--inner-meta", str(d / "meta.json"), "--partner-src", str(leg(d, "P", 2)),
           "--extra", f"A:0.2:{leg(d, 'A', 3, a, 'HX48_WORK')}:HX48_WORK:hx48", "--diag-encode", "--out", str(out)]
    return cmd + (["--total-h", str(total_h)] if total_h is not None else [])


def build(d, case, total_h=None, a="sleep"):
    fake = fake_root(d, case)
    out = fake / "kernel"
    subprocess.run(build_cmd(d, fake, out, total_h, a), check=True, capture_output=True)
    work = fake / "working"
    work.mkdir()
    return out / "joint.py", work


def run_forwarded(d, case, total_h, elapsed_after_p, min_leg_s=None):
    """main() with the wrapper clock set to `elapsed_after_p` s since start as soon as leg P returns"""
    joint, work = build(d, case, total_h)
    mod = load_module(joint, work)
    orig_run_leg = mod.run_leg

    def run_leg(name, *args, **kw):
        r = orig_run_leg(name, *args, **kw)
        if name == "P":
            mod.T0 = time.time() - elapsed_after_p
        return r
    mod.run_leg = run_leg
    if min_leg_s is not None:
        mod.MIN_LEG_S = min_leg_s
    mod.main()
    sub, rep = read(work)
    return mod, rep


def main():
    checks = {}
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        fake = fake_root(d, "render")
        texts = {}
        for key, h in (("default", None), ("8.6", 8.6), ("8.8", 8.8)):
            out = fake / f"k_{key}"
            subprocess.run(build_cmd(d, fake, out, h), check=True, capture_output=True)
            texts[key] = (out / "joint.py").read_text()
        checks["default build == explicit --total-h 8.6"] = texts["default"] == texts["8.6"]
        checks["default renders exactly `TOTAL_S = 8.6 * 3600`"] = (
            [ln for ln in texts["default"].splitlines() if ln.startswith("TOTAL_S")] == ["TOTAL_S = 8.6 * 3600"])
        diff = [(x, y) for x, y in zip(texts["default"].splitlines(), texts["8.8"].splitlines()) if x != y]
        checks["--total-h 8.8: only the TOTAL_S line differs"] = (
            len(texts["default"].splitlines()) == len(texts["8.8"].splitlines())
            and diff == [("TOTAL_S = 8.6 * 3600", "TOTAL_S = 8.8 * 3600")])

        for h in ("0.5", "8.9", "9", "-1"):
            out = d / f"bad_{h}"
            r = subprocess.run(build_cmd(d, fake, out, h), capture_output=True, text=True)
            checks[f"--total-h {h} rejected, no joint.py"] = (r.returncode != 0 and not (out / "joint.py").exists()
                                                             and "outside 1-8.85 h" in r.stderr)
        for h, line in (("1", "TOTAL_S = 1.0 * 3600"), ("8.85", "TOTAL_S = 8.85 * 3600")):
            out = d / f"edge_{h}"
            r = subprocess.run(build_cmd(d, fake, out, h), capture_output=True, text=True)
            checks[f"--total-h {h} accepted -> `{line}`"] = (r.returncode == 0
                                                            and line in (out / "joint.py").read_text().splitlines())

        # A starts with 8.6 h - elapsed - 300 s = 4 s of budget at 8.6 h, 724 s at 8.8 h; it sleeps A_SLEEP_S
        elapsed = 8.6 * 3600 - 300 - 4
        mod, rep = run_forwarded(d, "to_86", 8.6, elapsed, min_leg_s=0)
        checks["8.6 h: TOTAL_S == 8.6 * 3600"] = mod.TOTAL_S == 8.6 * 3600
        checks["8.6 h: A times out with ~4 s budget -> diag code 3"] = (
            rep["A"].get("reason") == "timeout" and rep["A"]["budget_s"] <= 4 and rep["diag_code"] == 3
            and rep["weights_used"] == {"V": 0.5, "P": 0.5})
        mod, rep = run_forwarded(d, "to_88", 8.8, elapsed, min_leg_s=0)
        checks["8.8 h: same clock, A finishes with ~724 s budget -> diag code 0"] = (
            "error" not in rep["A"] and 715 <= rep["A"]["budget_s"] <= 724 and rep["A"]["s"] >= A_SLEEP_S
            and rep["diag_code"] == 0 and rep["weights_used"] == {"V": 0.4, "P": 0.4, "A": 0.2})

        # real MIN_LEG_S (0.5 h): 1440 s left at 8.6 h -> skipped; 2160 s left at 8.8 h -> runs
        elapsed = 8.6 * 3600 - 1440
        mod, rep = run_forwarded(d, "sk_86", 8.6, elapsed)
        checks["8.6 h, 1440 s left < MIN_LEG_S: A skipped -> diag code 4"] = (
            mod.MIN_LEG_S == 1800 and rep["A"].get("reason") == "skipped" and rep["diag_code"] == 4)
        mod, rep = run_forwarded(d, "sk_88", 8.8, elapsed)
        checks["8.8 h, 2160 s left: A runs -> diag code 0"] = (
            "error" not in rep["A"] and 1850 <= rep["A"]["budget_s"] <= 1860 and rep["diag_code"] == 0)
    for k, v in checks.items():
        print("PASS" if v else "FAIL", k)
    sys.exit(1 if not all(checks.values()) else 0)


if __name__ == "__main__":
    main()
