"""Andrey's rsna-knee-r6-r7-blur-oof (2026-10-20) made runnable in a fake /kaggle tree: /kaggle/ and '/tmp paths are
remapped to ROOT in the wrapper and in both embedded legs (LEG_SRC, re-packed), and MODELS is set to one model so R6
and R7 run as two processes on two GPUs (his own "two sessions" split). Nothing else changes; checked below.

  python -I qa/andrey/prep_r67oof.py SRC ROOT MODEL OUT
"""
import ast
import base64
import sys
import zlib

src, root, model, out = sys.argv[1:]
assert model in ("r6", "r7")
text = open(src, encoding="utf-8").read()


def remap(s):
    n = s.count("/kaggle/")
    s = s.replace("/kaggle/", root + "/kaggle/").replace("'/tmp", f"'{root}/tmp")
    assert s.count(root + "/kaggle/") == n
    return s


def pack(s):
    return base64.b64encode(zlib.compress(s.encode("utf-8"), 9)).decode("ascii")


legs = next(n.value for n in ast.parse(text).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LEG_SRC")
new = text
for k, v in zip(legs.keys, legs.values):
    leg = zlib.decompress(base64.b64decode(v.value)).decode("utf-8")
    assert new.count(v.value) == 1, k.value
    new = new.replace(v.value, pack(remap(leg)))
old_models = "MODELS = ['r6', 'r7']\n"
assert new.count(old_models) == 1 and new.count("GO = True\n") == 1
new = remap(new.replace(old_models, f"MODELS = ['{model}']\n"))
open(out, "w", encoding="utf-8", newline="\n").write(new)
print("prepared", out, "model", model)
