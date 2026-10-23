"""Build qa/joint/kernel_joint/: our v3stack kernel + the partner's inference code (qa/joint, 2026-10-14).

The inner kernel is a built qa/v2stack/build_kernel.py output (default the v3stack one that scored 0.957), embedded
verbatim except its start time, which is taken from the joint wrapper (JOINT_T0) so both legs share one 9 h budget.
The partner source is a .py run as is, or an .ipynb whose code cells are joined; notebook-only lines (`!`, `%`) are
commented out and listed in the build output.  `--fake ROOT` rewrites /kaggle/ and '/tmp paths for the gold-58 test;
the inner kernel must then be one built with the same --fake.
"""
import argparse
import base64
import hashlib
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INNER_T0 = "T0 = time.time()\n"
INNER_T0_PATCH = "T0 = float(os.environ.get('JOINT_T0') or time.time())  # [joint] shared 9 h budget\n"

ap = argparse.ArgumentParser()
ap.add_argument("--partner-src", required=True, help="partner inference .py or .ipynb")
ap.add_argument("--partner-datasets", nargs="+", default=[], help="Kaggle datasets the partner code reads")
ap.add_argument("--inner", default=str(ROOT / "qa/v3stack/kernel_blend/blend_v2stack.py"))
ap.add_argument("--inner-meta", default=str(ROOT / "qa/v3stack/kernel_blend/kernel-metadata.json"),
                help="its kernel-metadata.json: dataset sources and docker image are carried over")
ap.add_argument("--w-partner", type=float, default=0.5)
ap.add_argument("--partner-first", action="store_true")
ap.add_argument("--partner-max-h", type=float, default=8.0)
ap.add_argument("--kernel-id", default="sroziews/rsna-kad-joint")
ap.add_argument("--docker-image", help="override the inner kernel's pinned image")
ap.add_argument("--partner-fast-glob", action="store_true",
                help="rewrite the partner's glob('/kaggle/input/**/NAME', recursive=True) into a walk that skips the "
                     "competition DICOM tree")
ap.add_argument("--extra", action="append", default=[], metavar="NAME:W:SRC[:WORK_ENV:SUBDIR]",
                help="extra leg run after V and P: a .py run unchanged, rank weight W of the final blend; WORK_ENV names "
                     "the env var that sets its output folder (WORK/SUBDIR), else it writes WORK/submission.csv")
ap.add_argument("--extra-max-h", type=float, default=3.0)
ap.add_argument("--extra-datasets", nargs="+", default=[], help="Kaggle datasets the extra legs read")
ap.add_argument("--require-all-legs", action="store_true",
                help="diagnostic: the kernel fails (no submission.csv) if any leg is dropped, instead of falling back")
ap.add_argument("--diag-encode", action="store_true",
                help="diagnostic: write the legs' outcome into the score by blanking findings (joint_template DIAG_ENCODE)")
ap.add_argument("--total-h", type=float, default=8.6,
                help="joint budget in hours from the wrapper's start (the last leg ends by TOTAL - 300 s); Kaggle stops the "
                     "notebook at 9 h, container start comes before the wrapper's clock; max 8.85")
ap.add_argument("--fake", help="test tree root replacing /kaggle and /tmp")
ap.add_argument("--out", default=str(HERE / "kernel_joint"))
a = ap.parse_args()
if not 1 <= a.total_h <= 8.85:
    ap.error(f"--total-h {a.total_h}: outside 1-8.85 h")


def fake(text):
    if not a.fake:
        return text
    return text.replace("/kaggle/", a.fake + "/").replace("'/tmp", f"'{a.fake}/tmp")


inner = Path(a.inner).read_text()
assert inner.count(INNER_T0) == 1, "inner kernel start-time line not found exactly once"
inner = inner.replace(INNER_T0, INNER_T0_PATCH)
src = Path(a.partner_src)
raw = src.read_bytes()
if src.suffix == ".ipynb":
    cells = ["".join(c["source"]) for c in json.loads(raw)["cells"] if c["cell_type"] == "code"]
    lines = []
    for ln in "\n\n".join(cells).splitlines():
        if ln.lstrip().startswith(("!", "%")):
            print("  [joint] commented out:", ln.strip()[:120])
            ln = "# [joint] " + ln
        lines.append(ln)
    partner = "\n".join(lines) + "\n"
else:
    partner = raw.decode()
if a.partner_fast_glob:
    # Each recursive glob walks the whole competition mount (train + test DICOMs): 703 + 153 s of the partner's 922 s
    # on the 3-study public run (2026-10-15).  The replacement is one expression, so it also works inside code the
    # partner writes out as a string literal (quotes there are \'), and keeps glob's list type for the [0] after it.
    # Walk order differs from glob's, so a NAME present in two attached datasets could resolve to the other copy.
    pat = re.compile(r"glob\.glob\((\\?')/kaggle/input/\*\*/([^'\\*/]+)\1, recursive=True\)")

    def walk_expr(m):
        q, name = m.group(1), m.group(2)
        return (f"[next(r + {q}/{name}{q} for r, d, fs in os.walk({q}/kaggle/input{q}) if not d.__setitem__("
                f"slice(None), [x for x in d if x != {q}rsna-knee-abnormality-detection{q}]) and {q}{name}{q} in fs)]")

    print("  [joint] fast glob:", [m.group(2) for m in pat.finditer(partner)])
    partner, n_glob = pat.subn(walk_expr, partner)
    assert n_glob and "/kaggle/input/**" not in partner, "partner recursive glob left unrewritten"
partner = fake(partner)


def packed(s):
    # Kaggle rejects script kernels over ~1 MB (HTTP 400 on push): plain base64 of v3stack + partner was 1.19 MB
    return base64.b64encode(zlib.compress(s.encode(), 9)).decode()


extra_legs, extra_src = [], {}
for spec in a.extra:
    name, w, path, *rest = spec.split(":")
    assert name not in ("V", "P") and name not in extra_src and len(rest) in (0, 2), spec
    extra_src[name] = fake(Path(path).read_text())
    extra_legs.append({"name": name, "w": float(w), "max_s": round(a.extra_max_h * 3600),
                       "work_env": rest[0] if rest else "", "subdir": rest[1] if rest else "",
                       "src": packed(extra_src[name])})
assert sum(x["w"] for x in extra_legs) < 1


text = (fake((HERE / "joint_template.py").read_text())
        .replace("@@INNER_MD5@@", hashlib.md5(inner.encode()).hexdigest())
        .replace("@@PARTNER_NAME@@", src.name)
        .replace("@@PARTNER_SHA@@", hashlib.sha256(raw).hexdigest())
        .replace("@@W_PARTNER@@", repr(a.w_partner))
        .replace("@@PARTNER_FIRST@@", repr(a.partner_first))
        .replace("@@PARTNER_MAX_S@@", repr(round(a.partner_max_h * 3600)))
        .replace("@@EXTRA_LEGS@@", repr(extra_legs))
        .replace("@@REQUIRE_ALL@@", repr(a.require_all_legs))
        .replace("@@DIAG_ENCODE@@", repr(a.diag_encode))
        .replace("@@TOTAL_H@@", repr(a.total_h))
        .replace("@@INNER_B64@@", packed(inner))
        .replace("@@PARTNER_B64@@", packed(partner)))
assert "@@" not in text.replace(packed(inner), "").replace(packed(partner), "")
inner_meta = json.loads(Path(a.inner_meta).read_text())
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "joint.py").write_text(text)
(out / "partner_src.py").write_text(partner)  # readable copy for review; the kernel embeds it
for name, s in extra_src.items():
    (out / f"extra_{name}_src.py").write_text(s)
meta = {
    "id": a.kernel_id,
    "title": a.kernel_id.split("/")[1],
    "code_file": "joint.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": list(dict.fromkeys(inner_meta["dataset_sources"] + a.partner_datasets + a.extra_datasets)),
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "kernel_sources": [],
    "model_sources": [],
    "docker_image": a.docker_image or inner_meta["docker_image"],
    "machine_shape": "NvidiaTeslaT4",
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print("built", out / "joint.py", len(text), "bytes; inner md5", hashlib.md5(inner.encode()).hexdigest(),
      "; partner", src.name, "; joint md5", hashlib.md5(text.encode()).hexdigest())
