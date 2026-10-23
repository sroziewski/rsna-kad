"""B-Pblur (2026-10-20): Andrey's j3-stack-blur kernel (team best, LB 0.961) with the Qile leg blurred the way our gold
run measured it: Gaussian sigma 0.65 px on every view after the warp (qa/blur/patch_legs.patch_qile_src, the patch of
the 0.960 probe 57405419) instead of B's 0.8 px on the 120 mm volume of MTH only (HIRES_BLUR_PX set to 0, so MTH is
not blurred twice). Gold-58 screen (.tmp/research_gold/variants.py): B 0.9504 -> 0.9515, P(>0) 0.81; B's stacker
there still saw unblurred members. Everything else in B stays byte-identical (checked below).

  python qa/bblur/build_bpblur.py --src B.py --meta kernel-metadata.json --out DIR [--sigma 0.65]
"""
import argparse
import ast
import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path

assert sys.version_info >= (3, 8), "ast string nodes need .value (Python >= 3.8)"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blur"))
from patch_legs import once, patch_qile_src, replace_literal  # noqa: E402
from patch_r7 import dict_value_pick  # noqa: E402

HIRES_OLD = "R7_BLUR_MM, HIRES_BLUR_PX = 0.288, 0.8\n"
HIRES_NEW = "R7_BLUR_MM, HIRES_BLUR_PX = 0.288, 0.0   # [bpblur] MTH volume blur off: every Qile view is blurred after the warp\n"
KERNEL_ID = "sroziews/rsna-kad-bpblur"


def src_dict(text):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SRC":
            return {k.value: v.value for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError("SRC not found")


def unpack(b64):
    return zlib.decompress(base64.b64decode(b64)).decode("utf-8")


def pack(text):
    return base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")


def build(text, sigma):
    assert sigma > 0, "sigma must be > 0"
    ours = unpack(src_dict(text)["ours"])
    new_ours = patch_qile_src(ours, sigma)
    out = replace_literal(text, dict_value_pick(text, "SRC", "ours"), lambda v: pack(new_ours))
    out = once(out, HIRES_OLD, HIRES_NEW)
    before, after = src_dict(text), src_dict(out)
    assert unpack(after["ours"]) == new_ours and list(before) == list(after)
    assert all(before[k] == after[k] for k in before if k != "ours")
    changed = [i for i, (x, y) in enumerate(zip(text.splitlines(), out.splitlines())) if x != y]
    assert len(text.splitlines()) == len(out.splitlines()) and len(changed) == 2, changed
    return out, ours, new_ours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Andrey's rsna-knee-j3-stack-blur.py (kernels pull)")
    ap.add_argument("--meta", required=True, help="its kernel-metadata.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=0.65)
    a = ap.parse_args()
    text = Path(a.src).read_text(encoding="utf-8")
    out, ours, new_ours = build(text, a.sigma)
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta.update(id=KERNEL_ID, title=KERNEL_ID.split("/")[1], code_file="bpblur.py")
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "bpblur.py").write_bytes(out.encode("utf-8"))
    (d / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    (d / "run_ours_patched.py").write_bytes(new_ours.encode("utf-8"))   # readable copy for review; the kernel embeds it
    md5 = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()
    print(f"sigma {a.sigma}: B {md5(text)} -> bpblur {md5(out)}; run_ours {md5(ours)} -> {md5(new_ours)}")


if __name__ == "__main__":
    main()
