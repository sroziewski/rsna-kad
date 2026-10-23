"""Checks of qa/blur/kernel_joint3blurall2 and of build_joint.py's default output after --total-h (qa/blur, 2026-10-20).

Rebuilds joint3blurall, joint3blur and joint3blurall2 with their build commands into .tmp/test_blurall2/ and compares
them file by file with the kernels in qa/; then decodes the legs embedded in joint3blurall2's joint.py.
    python qa/blur/test_blurall2.py
"""
import ast
import base64
import hashlib
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = ROOT / ".tmp/test_blurall2"
FILES = ("joint.py", "kernel-metadata.json", "partner_src.py", "extra_A_src.py")
INNER_MD5 = "667576f676eef7143b5b4980846c3e6e"
T = "qa/blur/src_s0.65"
PARTNER = ["--partner-src", f"{T}/qile_blur.ipynb", "--partner-datasets",
           "qilejiang76/rsna-knee-lineage-g1-mth-mtr-olp-20261013", "qilejiang76/rsna-knee-single-models-20261007"]
EXTRA_DS = ["--extra-datasets", "andreytikhomirov/rsna-knee-native-hx48-weights",
            "andreytikhomirov/rsna-knee-landmark-detector"]
BUILDS = {
    "kernel_joint3blurall": ["--inner", "qa/blur/v_inner_s0.288/blend_v2stack.py", *PARTNER,
                             "--extra", f"A:0.2:{T}/hx48_blur.py:HX48_WORK:hx48", *EXTRA_DS,
                             "--require-all-legs", "--kernel-id", "sroziews/rsna-kad-joint3blurall"],
    "kernel_joint3blur": [*PARTNER, "--extra", f"A:0.2:{T}/hx48_blur.py:HX48_WORK:hx48", *EXTRA_DS,
                          "--require-all-legs", "--kernel-id", "sroziews/rsna-kad-joint3blur"],
    "kernel_joint3blurall2": ["--inner", "qa/blur/v_inner_s0.288/blend_v2stack.py", *PARTNER,
                              "--extra", "A:0.2:qa/andrey/hx48_ensemble_v6.py:HX48_WORK:hx48", *EXTRA_DS,
                              "--diag-encode", "--total-h", "8.8", "--kernel-id", "sroziews/rsna-kad-joint3blurall2"],
}


def md5(b):
    return hashlib.md5(b).hexdigest()


def unpack(s):
    return zlib.decompress(base64.b64decode(s)).decode()


def constants(text):
    """top-level NAME = <literal> assignments of a joint.py"""
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def main():
    checks = {}
    for kernel, args in BUILDS.items():
        out = SCRATCH / kernel
        subprocess.run([sys.executable, "qa/joint/build_joint.py", *args, "--out", str(out)], cwd=ROOT, check=True,
                       capture_output=True)
        for f in FILES:
            new, old = (out / f).read_bytes(), (HERE / kernel / f).read_bytes()
            print(f"  {kernel}/{f}: rebuilt {md5(new)}  qa {md5(old)}")
            checks[f"{kernel}/{f} rebuilt byte-identical"] = new == old
        lines = [ln for ln in (out / "joint.py").read_text().splitlines() if ln.startswith("TOTAL_S")]
        want = "TOTAL_S = 8.8 * 3600" if kernel.endswith("2") else "TOTAL_S = 8.6 * 3600"
        checks[f"{kernel}: {want}"] = lines == [want]
    checks["joint3blurall joint.py md5 def6ff6a..."] = md5((HERE / "kernel_joint3blurall/joint.py").read_bytes()
                                                           ).startswith("def6ff6a")
    checks["joint3blur joint.py md5 e162d766..."] = md5((HERE / "kernel_joint3blur/joint.py").read_bytes()
                                                        ).startswith("e162d766")

    k2, k1, kb = (HERE / k for k in ("kernel_joint3blurall2", "kernel_joint3blurall", "kernel_joint3blur"))
    text = (k2 / "joint.py").read_text()
    c, c1, cb = constants(text), constants((k1 / "joint.py").read_text()), constants((kb / "joint.py").read_text())
    checks["REQUIRE_ALL False, DIAG_ENCODE True, W_PARTNER 0.5"] = (
        c["REQUIRE_ALL"] is False and c["DIAG_ENCODE"] is True and c["W_PARTNER"] == 0.5)
    checks["TOTAL_S = 8.8 * 3600"] = re.findall(r"^TOTAL_S = .*$", text, re.M) == ["TOTAL_S = 8.8 * 3600"]
    legs = c["EXTRA_LEGS"]
    checks["one extra leg A: w 0.2, max_s 10800, HX48_WORK, hx48"] = (
        len(legs) == 1 and {k: v for k, v in legs[0].items() if k != "src"}
        == {"name": "A", "w": 0.2, "max_s": 10800, "work_env": "HX48_WORK", "subdir": "hx48"})
    inner = unpack(c["INNER_SRC"])
    print(f"  inner md5 blurall2 {md5(inner.encode())}  blurall {md5(unpack(c1['INNER_SRC']).encode())}")
    checks[f"inner md5 {INNER_MD5} == joint3blurall's inner"] = (
        md5(inner.encode()) == INNER_MD5 == md5(unpack(c1["INNER_SRC"]).encode()))
    # the blur sits in the inner's base leg (base64 BASE_SRC), inside its SOURCES dict, i.e. after a literal "\n"
    base = base64.b64decode(constants(inner)["BASE_SRC"]).decode()
    checks["inner's base leg == v_inner_s0.288/base_v2.py, has K1_BLUR_MM = 0.288"] = (
        base == (HERE / "v_inner_s0.288/base_v2.py").read_text()
        and re.search(r"K1_BLUR_MM = 0\.288(?![0-9])", base) is not None)
    checks["inner md5 in the docstring"] = f"md5 {INNER_MD5}" in text
    partner = unpack(c["PARTNER_SRC"])
    checks["partner == joint3blur's (embedded and partner_src.py)"] = (
        partner == unpack(cb["PARTNER_SRC"]) == (kb / "partner_src.py").read_text() == (k2 / "partner_src.py").read_text())
    a_src = unpack(legs[0]["src"])
    v6 = (ROOT / "qa/andrey/kernel_joint3v6/extra_A_src.py").read_text()
    checks["A src == kernel_joint3v6/extra_A_src.py == extra_A_src.py, no _k1_blur"] = (
        a_src == v6 == (k2 / "extra_A_src.py").read_text() and "_k1_blur" not in a_src)
    m2, m1 = (json.loads((k / "kernel-metadata.json").read_text()) for k in (k2, k1))
    checks["kernel-metadata == joint3blurall's except id/title"] = (
        {k: v for k, v in m2.items() if k not in ("id", "title")} == {k: v for k, v in m1.items() if k not in ("id", "title")}
        and m2["id"] == "sroziews/rsna-kad-joint3blurall2" and m2["title"] == "rsna-kad-joint3blurall2")
    for k, v in checks.items():
        print("PASS" if v else "FAIL", k)
    sys.exit(1 if not all(checks.values()) else 0)


if __name__ == "__main__":
    main()
