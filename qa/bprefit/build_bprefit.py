"""bprefit (2026-10-21): bpblur (LB 0.961) with stacker5 refitted by Andrey's fit_stacker5.py on out-of-fold predictions
made with bpblur's own Qile path (0.65 px after the warp, qa/bprefit/qoof_driver.py). Only the `STK = json.loads(...)`
line changes; features, members and the blend are bpblur's.

  python qa/bprefit/build_bprefit.py --bpblur bpblur.py --stk stacker5_bprefit.json --meta kernel-metadata.json --out DIR
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

assert sys.version_info >= (3, 8), "ast string nodes need .value (Python >= 3.8)"
KERNEL_ID = "sroziews/rsna-kad-bprefit"
PREFIX = "STK = json.loads("


def stk_line(text):
    lines = [i for i, l in enumerate(text.split("\n")) if l.startswith(PREFIX)]
    assert len(lines) == 1, f"{len(lines)} STK lines"
    return lines[0]


def parse_stk(line):
    call = ast.parse(line).body[0].value
    return json.loads(call.args[0].value)


def build(bp, new):
    lines = bp.split("\n")
    i = stk_line(bp)
    old = parse_stk(lines[i])
    assert new["features"] == old["features"] and new["members"] == old["members"], "feature order must stay"
    assert set(new["lr"]) == set(old["lr"]) == set(new["ranklr"]), "findings differ"
    lines[i] = PREFIX + repr(json.dumps(new, separators=(",", ":"))) + ")"
    out = "\n".join(lines)
    assert parse_stk(out.split("\n")[stk_line(out)]) == new
    a, b = bp.split("\n"), out.split("\n")
    assert len(a) == len(b) and [k for k in range(len(a)) if a[k] != b[k]] == [i], "only the STK line may change"
    return out, old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpblur", required=True)
    ap.add_argument("--stk", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bp = Path(a.bpblur).read_text(encoding="utf-8")
    new = json.loads(Path(a.stk).read_text(encoding="utf-8"))
    out, old = build(bp, new)
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta.update(id=KERNEL_ID, title=KERNEL_ID.split("/")[1], code_file="bprefit.py")
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "bprefit.py").write_bytes(out.encode("utf-8"))
    (d / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    md5 = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()
    dl = max(abs(x - y) for f in new["lr"] for x, y in zip(new["lr"][f], old["lr"][f]))
    print(f"bpblur {md5(bp)} -> bprefit {md5(out)}; max |d lr| vs old {dl:.3f}")


if __name__ == "__main__":
    main()
