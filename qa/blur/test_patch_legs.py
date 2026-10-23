"""Tests for qa/blur/patch_legs.py (run on the server, CPU only):

  CUDA_VISIBLE_DEVICES= python -m pytest -q qa/blur/test_patch_legs.py

The real-input tests read QILE_NB / HX48 (env overrides) and skip when they are absent. The embedded blur functions are
taken out of the patched source text (only the K1_BLUR_SIGMA assignment and def _k1_blur are exec'd), never by running
the teammates' scripts."""
import ast
import difflib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("patch_legs", HERE / "patch_legs.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
QILE_NB = Path(os.environ.get("QILE_NB", ROOT / ".tmp/qile/v3nb/rsna-knee-ours-mtr1-mth4-olp5.ipynb"))
HX48 = Path(os.environ.get("HX48", ROOT / "qa/andrey/hx48_ensemble_v6.py"))
SCRATCH = Path(os.environ.get("BLUR_TEST_SCRATCH", ROOT / ".tmp/blur/test"))
SIGMAS = [0.5, 0.65, 0.8, 1.3, 0.6]

need_nb = pytest.mark.skipif(not QILE_NB.exists(), reason=f"{QILE_NB} missing")
need_hx = pytest.mark.skipif(not HX48.exists(), reason=f"{HX48} missing")


def files_dict(text):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FILES":
            return ast.literal_eval(node.value)
    raise AssertionError("FILES not found")


def cell_src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def target_literal(src, needle):
    """(node, value) of the single str constant containing needle, parsed from src (no magics in these cells)."""
    nodes = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and needle in n.value]
    assert len(nodes) == 1, len(nodes)
    return nodes[0], nodes[0].value


def only_inserts(old, new, allowed_replace=()):
    """Line diff old -> new; asserts it is inserts only, plus the replace pairs listed in allowed_replace."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    inserted, replaced = [], []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "insert":
            inserted.append("".join(b[j1:j2]))
        elif tag == "replace":
            replaced.append(("".join(a[i1:i2]), "".join(b[j1:j2])))
        else:
            raise AssertionError(f"unexpected {tag} at old lines {i1}:{i2}: {a[i1:i2]!r}")
    assert sorted(replaced) == sorted(allowed_replace), replaced
    return inserted


def extract_blur(src, extra_ns):
    """exec only `K1_BLUR_SIGMA = ...` and `def _k1_blur` from src; returns the function."""
    tree = ast.parse(src)
    parts = [n for n in tree.body if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "K1_BLUR_SIGMA")
             or (isinstance(n, ast.FunctionDef) and n.name == "_k1_blur")]
    assert len(parts) == 2, [type(p).__name__ for p in parts]
    ns = dict(extra_ns)
    exec(compile("\n\n".join(ast.get_source_segment(src, p) for p in parts), "<k1_blur>", "exec"), ns)
    return ns["_k1_blur"], ns["K1_BLUR_SIGMA"]


def module_names(src):
    """Names bound at module level by imports."""
    out = set()
    for n in ast.parse(src).body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return out


# ---------------------------------------------------------------- fixtures on the real inputs

@pytest.fixture(scope="module")
def hx_text():
    return HX48.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def nb_text():
    return QILE_NB.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def infer_fast(hx_text):
    return files_dict(hx_text)["infer_fast.py"]


@pytest.fixture(scope="module")
def ours_infer(nb_text):
    return target_literal(cell_src(json.loads(nb_text)["cells"][0]), pl.QILE_ANCHOR)[1]


# ---------------------------------------------------------------- 1 + 5: hx48 diff is only the blur

@need_hx
@pytest.mark.parametrize("sigma", [0, 0.65])
def test_hx48_only_files_line_and_infer_fast_change(hx_text, sigma):
    new = pl.patch_hx48(hx_text, sigma)
    a, b = hx_text.splitlines(keepends=True), new.splitlines(keepends=True)
    assert len(a) == len(b)
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    files_line = next(i for i, ln in enumerate(a) if ln.startswith("FILES = "))
    assert changed == [files_line]
    fa, fb = files_dict(hx_text), files_dict(new)
    assert list(fa) == list(fb)
    assert [k for k in fa if fa[k] != fb[k]] == ["infer_fast.py"]
    # the bytes around the literal are untouched
    node = [n for n in ast.parse(hx_text).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FILES"][0]
    v = node.value.values[[k.value for k in node.value.keys].index("infer_fast.py")]
    la, lb = a[files_line].encode(), b[files_line].encode()
    assert lb[:v.col_offset] == la[:v.col_offset]
    tail = la[v.end_col_offset:]
    assert lb.endswith(tail)


@need_hx
def test_hx48_infer_fast_diff_is_blur_only(infer_fast):
    new = pl.patch_hx_src(infer_fast, 0.65)
    anchor_line = next(ln for ln in infer_fast.splitlines(keepends=True) if pl.HX_ANCHOR in ln)
    ins = only_inserts(infer_fast, new, [(anchor_line, anchor_line.replace(pl.HX_ANCHOR, pl.HX_NEW))])
    assert "".join(ins).strip() == pl.HX_FN.format(sigma=0.65).strip()
    assert new.count("_k1_blur(") == 2
    assert {"math", "torch"} <= module_names(new)  # names _k1_blur needs at call time


@need_hx
@pytest.mark.parametrize("sigma", [0, 0.65])
def test_hx48_compiles_and_literal_roundtrips(hx_text, infer_fast, sigma):
    new = pl.patch_hx48(hx_text, sigma)
    compile(new, "hx48_blur.py", "exec")
    got = files_dict(new)["infer_fast.py"]
    assert got == pl.patch_hx_src(infer_fast, sigma)
    compile(got, "infer_fast.py", "exec")


# ---------------------------------------------------------------- 1 + 5: notebook diff is only the blur

@need_nb
@pytest.mark.parametrize("sigma", [0, 0.65])
def test_nb_only_cell0_changes(nb_text, sigma):
    a, b = json.loads(nb_text), json.loads(pl.patch_qile_nb(nb_text, sigma))
    assert {k: v for k, v in a.items() if k != "cells"} == {k: v for k, v in b.items() if k != "cells"}
    assert len(a["cells"]) == len(b["cells"])
    for i, (ca, cb) in enumerate(zip(a["cells"], b["cells"])):
        assert {k: v for k, v in ca.items() if k != "source"} == {k: v for k, v in cb.items() if k != "source"}, i
        if i:
            assert cell_src(ca) == cell_src(cb), i
    sa, sb = cell_src(a["cells"][0]), cell_src(b["cells"][0])
    na, _ = target_literal(sa, pl.QILE_ANCHOR)
    nb_, _ = target_literal(sb, pl.QILE_ANCHOR)
    assert (na.lineno, na.col_offset) == (nb_.lineno, nb_.col_offset)
    ra, rb = sa.encode(), sb.encode()
    la = sa.splitlines(keepends=True)
    start = sum(len(x.encode()) for x in la[:na.lineno - 1]) + na.col_offset
    end = sum(len(x.encode()) for x in la[:na.end_lineno - 1]) + na.end_col_offset
    assert rb[:start] == ra[:start]
    assert rb.endswith(ra[end:])


@need_nb
def test_nb_literal_diff_is_inserts_only(ours_infer):
    new = pl.patch_qile_src(ours_infer, 0.65)
    ins = only_inserts(ours_infer, new)
    assert len(ins) == 2
    assert pl.QILE_FN.format(sigma=0.65) in ins[0]
    assert ins[1] == "        crops[(img, crop_mm)] = _k1_blur(crops[(img, crop_mm)])   # [blur probe]\n"
    # the blur line comes right after the anchor line
    assert pl.QILE_ANCHOR + ins[1] in new
    assert {"np", "cv2"} <= module_names(new)


@need_nb
@pytest.mark.parametrize("sigma", [0, 0.65])
def test_nb_compiles_and_literal_roundtrips(nb_text, ours_infer, sigma):
    b = json.loads(pl.patch_qile_nb(nb_text, sigma))
    for c in b["cells"]:
        if c["cell_type"] == "code":
            compile(cell_src(c), "cell", "exec")
    node, val = target_literal(cell_src(b["cells"][0]), pl.QILE_ANCHOR)
    assert val == pl.patch_qile_src(ours_infer, sigma)
    assert ast.literal_eval(ast.get_source_segment(cell_src(b["cells"][0]), node)) == val
    compile(val, "ours_infer.py", "exec")


@need_nb
@need_hx
def test_cli_writes_same_as_functions(nb_text, hx_text):
    out = SCRATCH / "cli"
    r = subprocess.run([sys.executable, str(HERE / "patch_legs.py"), "--sigma", "0.65", "--qile-nb", str(QILE_NB),
                        "--hx48", str(HX48), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "qile_blur.ipynb").read_text(encoding="utf-8") == pl.patch_qile_nb(nb_text, 0.65)
    assert (out / "hx48_blur.py").read_text(encoding="utf-8") == pl.patch_hx48(hx_text, 0.65)
    assert HX48.read_text(encoding="utf-8") == hx_text and QILE_NB.read_text(encoding="utf-8") == nb_text


# ---------------------------------------------------------------- 2 + 3: the embedded blur functions

def qile_blur(sigma):
    import cv2
    return extract_blur(pl.patch_qile_src(pl.QILE_IMPORT + "def decode():\n" + pl.QILE_ANCHOR, sigma), {"np": np, "cv2": cv2})


def hx_blur(sigma):
    import torch
    return extract_blur(pl.patch_hx_src(pl.HX_IMPORT + pl.HX_ANCHOR + "\n", sigma), {"torch": torch, "math": math})


def test_qile_sigma0_identity():
    f, s = qile_blur(0)
    assert s == 0.0
    a = np.random.default_rng(0).integers(0, 256, (3, 224, 224), dtype=np.uint8)
    assert f(a) is a


def test_hx_sigma0_identity():
    import torch
    f, s = hx_blur(0)
    assert s == 0.0
    x = torch.randint(0, 256, (1, 5, 3, 64, 64), dtype=torch.uint8)
    assert f(x) is x and x.dtype == torch.uint8


@need_nb
@need_hx
def test_blur_fn_from_real_sources_matches_template(ours_infer, infer_fast):
    import cv2, torch
    fq, sq = extract_blur(pl.patch_qile_src(ours_infer, 0.65), {"np": np, "cv2": cv2})
    fh, sh = extract_blur(pl.patch_hx_src(infer_fast, 0.65), {"torch": torch, "math": math})
    assert sq == sh == 0.65
    a = np.random.default_rng(1).integers(0, 256, (2, 96, 96), dtype=np.uint8)
    assert np.array_equal(fq(a), qile_blur(0.65)[0](a))
    x = torch.from_numpy(a).reshape(1, 1, 2, 96, 96)
    assert torch.equal(fh(x), hx_blur(0.65)[0](x))


@pytest.mark.parametrize("sigma", SIGMAS)
def test_qile_blur_matches_cv2(sigma):
    import cv2
    f, _ = qile_blur(sigma)
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (4, 224, 224), dtype=np.uint8)
    a[1] = 0
    a[2] = 255
    a[3, :, :112] = 0
    a[3, :, 112:] = 255
    got = f(a)
    assert got.dtype == np.uint8 and got.shape == a.shape and got is not a
    ref = np.stack([np.clip(np.rint(cv2.GaussianBlur(s.astype(np.float32), (0, 0), sigma,
                                                      borderType=cv2.BORDER_REFLECT_101)), 0, 255).astype(np.uint8) for s in a])
    assert np.array_equal(got, ref)
    assert np.array_equal(got[1], a[1]) and np.array_equal(got[2], a[2])
    assert not np.array_equal(got[0], a[0])


HX_DIFFS = {}


@pytest.mark.parametrize("shape", [(1, 5, 3, 64, 64), (2, 3, 1, 48, 80)])
@pytest.mark.parametrize("sigma", SIGMAS)
def test_hx_blur_matches_cv2(sigma, shape):
    import cv2, torch
    f, _ = hx_blur(sigma)
    rng = np.random.default_rng(3)
    xn = rng.integers(0, 256, shape, dtype=np.uint8)
    x = torch.from_numpy(xn)
    y = f(x)
    assert tuple(y.shape) == shape and y.dtype == torch.float32
    assert torch.equal(x, torch.from_numpy(xn))  # input not modified
    y2 = f(x.float())
    assert torch.allclose(y, y2)
    yn = y.numpy().reshape(-1, shape[-2], shape[-1])
    planes = xn.reshape(-1, shape[-2], shape[-1]).astype(np.float32)
    r = max(1, (int(round(8 * sigma + 1)) | 1) // 2)   # patch_legs: cv2.GaussianBlur's own float kernel size
    ref_auto = np.stack([cv2.GaussianBlur(p, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101) for p in planes])
    ref_same = np.stack([cv2.GaussianBlur(p, (2 * r + 1, 2 * r + 1), sigma, borderType=cv2.BORDER_REFLECT_101) for p in planes])
    d_auto, d_same = float(np.abs(yn - ref_auto).max()), float(np.abs(yn - ref_same).max())
    HX_DIFFS[(sigma, shape)] = (r, d_auto, d_same)
    print(f"\nsigma {sigma} shape {shape}: torch r={r} (k={2 * r + 1}), cv2 auto k={int(round(sigma * 8 + 1)) | 1}; "
          f"max|torch-cv2 auto|={d_auto:.3g}, max|torch-cv2 k=2r+1|={d_same:.3g}")
    assert d_same < 1e-3, d_same
    # the torch kernel uses cv2's own float32 size rule (k = round(8 sigma + 1) | 1), so leg A blurs exactly as leg P
    assert int(round(sigma * 8 + 1)) | 1 == 2 * r + 1
    assert d_auto < 1e-3, d_auto


# ---------------------------------------------------------------- 4: replace_literal robustness

def up(v):
    return v + "|é"


def test_replace_literal_non_ascii_before_on_same_line():
    text = 'x = {"ä€😀": "ü"}; y = "TARGET"; z = "ñ"\nw = 1\n'
    out = pl.replace_literal(text, lambda n: n.value == "TARGET", up)
    assert out == 'x = {"ä€😀": "ü"}; y = ' + repr("TARGET|é") + '; z = "ñ"\nw = 1\n'


def test_replace_literal_non_ascii_lines_before_and_multiline_literal():
    text = '# zażółć gęślą jaźń\na = "€"\nb = f("ą", """TAR\nGET ł\n""") ; c = "ö"\n'
    out = pl.replace_literal(text, lambda n: "TAR" in n.value, up)
    assert out == '# zażółć gęślą jaźń\na = "€"\nb = f("ą", ' + repr("TAR\nGET ł\n|é") + ') ; c = "ö"\n'


def test_replace_literal_crlf():
    text = 'a = "ä"\r\nb = "TARGET"\r\nc = 1\r\n'
    out = pl.replace_literal(text, lambda n: n.value == "TARGET", up)
    assert out == 'a = "ä"\r\nb = ' + repr("TARGET|é") + '\r\nc = 1\r\n'


def test_replace_literal_with_magic_lines():
    text = '!pip install -q x\n%matplotlib inline\n%%capture\nimport os\nopen("p", "w").write("TARGET ä")\n!ls -la\n'
    out = pl.replace_literal(text, lambda n: n.value.startswith("TARGET"), up)
    assert out == ('!pip install -q x\n%matplotlib inline\n%%capture\nimport os\nopen("p", "w").write('
                   + repr("TARGET ä|é") + ')\n!ls -la\n')


def test_replace_literal_value_with_lines_starting_bang_or_percent():
    # a triple-quoted literal whose own content has lines starting with ! or % (column 0); the value handed to patch()
    # and written back must be the real one
    text = 'open("p", "w").write("""TARGET\n%matplotlib inline\n!pip install x\n""")\n'
    out = pl.replace_literal(text, lambda n: n.value.startswith("TARGET"), lambda v: v)
    assert ast.literal_eval(out[len('open("p", "w").write('):-2]) == "TARGET\n%matplotlib inline\n!pip install x\n"


@pytest.mark.parametrize("hits", [0, 2])
def test_replace_literal_match_count_asserts(hits):
    text = 'a = "T"\nb = "T"\n' if hits == 2 else 'a = "x"\n'
    with pytest.raises(AssertionError):
        pl.replace_literal(text, lambda n: n.value == "T", up)


def test_once_missing_and_twice():
    with pytest.raises(AssertionError):
        pl.once("abc", "x", "y")
    with pytest.raises(AssertionError):
        pl.once("xx", "x", "y")
    assert pl.once("axb", "x", "y") == "ayb"


@pytest.mark.parametrize("which", ["import", "anchor"])
@pytest.mark.parametrize("count", [0, 2])
def test_patch_src_anchor_counts_assert(which, count):
    for imp, anchor, fn in [(pl.QILE_IMPORT, pl.QILE_ANCHOR, pl.patch_qile_src),
                            (pl.HX_IMPORT, pl.HX_ANCHOR + "\n", pl.patch_hx_src)]:
        parts = {"import": imp, "anchor": anchor}
        s = "".join(parts[k] * (count if k == which else 1) for k in ("import", "anchor"))
        with pytest.raises(AssertionError):
            fn(s, 0.65)


def make_nb(*sources):
    return json.dumps({"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                                  "source": s} for s in sources], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def test_patch_qile_nb_synthetic_and_counts():
    inner = pl.QILE_IMPORT + pl.QILE_ANCHOR
    cell = "!pip -q install x\nopen('/kaggle/working/ours_infer.py', 'w').write(" + repr(inner) + ")\n"
    other = "%time\nprint('ä')\n"
    out = json.loads(pl.patch_qile_nb(make_nb(cell, other), 0.5))
    s0 = cell_src(out["cells"][0])
    assert s0.startswith("!pip -q install x\nopen(")
    lit = s0[len("!pip -q install x\nopen('/kaggle/working/ours_infer.py', 'w').write("):-2]
    assert ast.literal_eval(lit) == pl.patch_qile_src(inner, 0.5)
    assert cell_src(out["cells"][1]) == other
    with pytest.raises(AssertionError):
        pl.patch_qile_nb(make_nb(other), 0.5)
    with pytest.raises(AssertionError):
        pl.patch_qile_nb(make_nb(cell, cell), 0.5)


def test_files_literal_missing():
    with pytest.raises(AssertionError):
        pl.files_literal("X = 1\n", "infer_fast.py")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
