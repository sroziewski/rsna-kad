"""bpsyn (2026-10-21): bpblur (LB 0.961) + the Synovitis stacker of Andrey's rsna-knee-flat4-syn-r6 (LB 0.961, one place
above B): after bpblur's final blend, Synovitis = (1 - share) rank(bpblur Synovitis) + share rank(stacker), share 0.25,
stacker = 0.5 rank(X @ lr) + 0.5 rank(X @ ranklr) on the z-scored percentile ranks of MTH / OLP / hx48 over all 12
findings (fitted by Andrey out-of-fold on 4,349 studies). B's own 12-finding stacker stays as it is, as in flat4 the
Synovitis stacker comes on top of the blend. The stacker JSON is copied as the literal of his source; the other 11
columns are bpblur's, unchanged.

  python qa/bblur/build_bpsyn.py --bpblur qa/bblur/kernel_bpblur/bpblur.py --flat4 FLAT4.py --meta META --out DIR
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

assert sys.version_info >= (3, 8), "ast string nodes need .value (Python >= 3.8)"
ANCHOR = "say('stacker5 blend: features', X.shape)\n"
KERNEL_ID = "sroziews/rsna-kad-bpsyn"
BLOCK = """\
# [bpsyn] the Synovitis stacker of andreytikhomirov/rsna-knee-flat4-syn-r6, JSON verbatim from its source
SYN = json.loads(@@LIT@@)
KEY_SYN = {'MT-hires': 'MTH', 'OL-plane': 'OLP', 'hx48': 'hx48'}
XS = np.stack([RK[KEY_SYN[f.split(' ', 1)[0]]][f.split(' ', 1)[1]] for f in SYN['features']], 1)
XS = (XS - np.array(SYN['mean'])) / np.array(SYN['std'])
stack_syn = 0.5 * prk(XS @ np.array(SYN['lr'])) + 0.5 * prk(XS @ np.array(SYN['ranklr']))
j_syn = labels.index(SYN['finding'])
out[:, j_syn] = (1 - SYN['share']) * prk(out[:, j_syn]) + SYN['share'] * prk(stack_syn)
say('[bpsyn] Synovitis stacker: features', XS.shape, 'share', SYN['share'])
"""


def stk_literal(text):
    """The source text of the string passed to json.loads on the `STK = ` line, and its parsed value."""
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "STK":
            arg = node.value.args[0]
            return ast.get_source_segment(text, arg), json.loads(arg.value)
    raise AssertionError("STK not found")


def build(bp, flat4):
    lit, syn = stk_literal(flat4)
    assert syn["finding"] == "Synovitis" and syn["share"] == 0.25 and len(syn["features"]) == 36, syn.get("note")
    assert {f.split(" ", 1)[0] for f in syn["features"]} == {"MT-hires", "OL-plane", "hx48"}
    assert bp.count(ANCHOR) == 1, "bpblur blend anchor not found exactly once"
    block = BLOCK.replace("@@LIT@@", lit)
    out = bp.replace(ANCHOR, block + ANCHOR)
    assert json.loads(ast.literal_eval(lit)) == syn
    a, b = bp.splitlines(), out.splitlines()
    k = a.index(ANCHOR.rstrip("\n"))
    assert b[:k] == a[:k] and b[k + len(block.splitlines()):] == a[k:], "only the block may be inserted"
    return out, syn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpblur", required=True, help="built bpblur.py (qa/bblur/build_bpblur.py)")
    ap.add_argument("--flat4", required=True, help="andreytikhomirov/rsna-knee-flat4-syn-r6 source (kernels pull)")
    ap.add_argument("--meta", required=True, help="bpblur's kernel-metadata.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bp = Path(a.bpblur).read_text(encoding="utf-8")
    out, _ = build(bp, Path(a.flat4).read_text(encoding="utf-8"))
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    meta.pop("id_no", None)
    meta.update(id=KERNEL_ID, title=KERNEL_ID.split("/")[1], code_file="bpsyn.py")
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "bpsyn.py").write_bytes(out.encode("utf-8"))
    (d / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    md5 = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()
    print(f"bpblur {md5(bp)} -> bpsyn {md5(out)}")


if __name__ == "__main__":
    main()
