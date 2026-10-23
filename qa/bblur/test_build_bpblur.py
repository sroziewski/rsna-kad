"""Tests for qa/bblur/build_bpblur.py (run on the server, CPU only):

  CUDA_VISIBLE_DEVICES= python -m pytest -q qa/bblur/test_build_bpblur.py

Andrey's B kernel (B_SRC / B_META, env overrides) is untrusted data: it is only parsed with ast and its SRC legs are only
zlib-decoded and compile()d, never run. The one function exec'd is our own QILE_FN template from qa/blur/patch_legs.py.
The real-input tests skip when B is absent."""
import ast
import difflib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_bpblur.py"
spec = importlib.util.spec_from_file_location("build_bpblur", BUILD)
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)
pl = sys.modules["patch_legs"]

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
B_DIR = ROOT / ".tmp/andrey/blurpull/rsna-knee-j3-stack-blur/code"
B_SRC = Path(os.environ.get("B_SRC", B_DIR / "rsna-knee-j3-stack-blur.py"))
B_META = Path(os.environ.get("B_META", B_DIR / "kernel-metadata.json"))
BUILT = Path(os.environ.get("BPBLUR_BUILT", HERE / "kernel_bpblur"))
SCRATCH = Path(os.environ.get("BPBLUR_TEST_SCRATCH", ROOT / ".tmp/test_bpblur"))
SIGMA = 0.65
BLUR_LINE = "        crops[(img, crop_mm)] = _k1_blur(crops[(img, crop_mm)])   # [blur probe]\n"

need_b = pytest.mark.skipif(not (B_SRC.exists() and B_META.exists()), reason=f"{B_SRC} missing")


def src_node(text):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SRC"]
    assert len(nodes) == 1, len(nodes)
    return nodes[0]


def src_values(text):
    return {k.value: v.value for k, v in zip(src_node(text).value.keys, src_node(text).value.values)}


def byte_span(text, node):
    lines = text.encode("utf-8").splitlines(keepends=True)
    start = sum(len(x) for x in lines[:node.lineno - 1]) + node.col_offset
    end = sum(len(x) for x in lines[:node.end_lineno - 1]) + node.end_col_offset
    return start, end


def ours_value_node(text):
    n = src_node(text)
    return n.value.values[[k.value for k in n.value.keys].index("ours")]


def hires_node(text):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
             and [e.id for e in n.targets[0].elts] == ["R7_BLUR_MM", "HIRES_BLUR_PX"]]
    assert len(nodes) == 1, len(nodes)
    return nodes[0]


def run_cli(*args):
    return subprocess.run([sys.executable, "-I", str(BUILD), *map(str, args)], capture_output=True, text=True)


def qile_blur(sigma):
    import cv2
    ns = {"np": np, "cv2": cv2}
    exec(compile(pl.QILE_FN.format(sigma=float(sigma)), "<QILE_FN>", "exec"), ns)
    return ns["_k1_blur"], ns["K1_BLUR_SIGMA"]


# ---------------------------------------------------------------- fixtures on the real inputs

@pytest.fixture(scope="module")
def b_text():
    return B_SRC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def built_text():
    return (BUILT / "bpblur.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def old_ours(b_text):
    return bb.unpack(src_values(b_text)["ours"])


@pytest.fixture(scope="module")
def new_ours(built_text):
    return bb.unpack(src_values(built_text)["ours"])


# ---------------------------------------------------------------- 1: only SRC['ours'] and the HIRES line change

@need_b
def test_built_equals_fresh_build(b_text, built_text):
    assert bb.build(b_text, SIGMA)[0] == built_text


@need_b
def test_top_level_statements_identical_except_src_and_hires(b_text, built_text):
    ta, tb = ast.parse(b_text).body, ast.parse(built_text).body
    assert len(ta) == len(tb)
    src_i = [i for i, n in enumerate(ta) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SRC"]
    hires_i = [i for i, n in enumerate(ta) if ast.dump(n) == ast.dump(hires_node(b_text))]
    assert len(src_i) == 1 and len(hires_i) == 1
    changed = [i for i, (x, y) in enumerate(zip(ta, tb)) if ast.dump(x) != ast.dump(y)]
    assert changed == sorted(src_i + hires_i)
    h = hires_node(built_text)
    assert ast.literal_eval(h.value) == (0.288, 0.0)
    assert ast.literal_eval(hires_node(b_text).value) == (0.288, 0.8)


@need_b
def test_src_keys_order_and_other_values_identical(b_text, built_text):
    a, b = src_values(b_text), src_values(built_text)
    assert list(a) == list(b)
    assert "ours" in a
    assert [k for k in a if a[k] != b[k]] == ["ours"]
    na, nb_ = src_node(b_text).value, src_node(built_text).value
    for k, va, vb in zip(a, na.values, nb_.values):
        if k != "ours":
            assert ast.get_source_segment(b_text, va) == ast.get_source_segment(built_text, vb), k


@need_b
def test_text_outside_two_spans_is_identical(b_text, built_text):
    ra, rb = b_text.encode("utf-8"), built_text.encode("utf-8")
    sa, ea = byte_span(b_text, ours_value_node(b_text))
    sb, eb = byte_span(built_text, ours_value_node(built_text))
    assert sa == sb and ra[:sa] == rb[:sb]
    old_rest, new_rest = ra[ea:], rb[eb:]
    ho, hn = bb.HIRES_OLD.encode(), bb.HIRES_NEW.encode()
    assert old_rest.count(ho) == 1 and new_rest.count(hn) == 1
    i, j = old_rest.index(ho), new_rest.index(hn)
    assert old_rest[:i] == new_rest[:j]
    assert old_rest[i + len(ho):] == new_rest[j + len(hn):]
    assert ast.literal_eval(rb[sb:eb].decode("utf-8")) == src_values(built_text)["ours"]


@need_b
def test_built_kernel_compiles(built_text):
    compile(built_text, "bpblur.py", "exec")


# ---------------------------------------------------------------- 2: the new SRC['ours']

@need_b
def test_new_ours_is_patch_of_old(old_ours, new_ours):
    assert new_ours == pl.patch_qile_src(old_ours, SIGMA)
    assert new_ours == (BUILT / "run_ours_patched.py").read_text(encoding="utf-8")
    compile(new_ours, "run_ours.py", "exec")
    compile(old_ours, "run_ours_orig.py", "exec")


@need_b
def test_new_ours_blur_line_after_single_warp(old_ours, new_ours):
    lines = new_ours.splitlines(keepends=True)
    blur = [i for i, ln in enumerate(lines) if "_k1_blur(crops[(img, crop_mm)])" in ln]
    warp = [i for i, ln in enumerate(lines) if "cv2.warpAffine(" in ln]
    assert len(blur) == 1 and len(warp) == 1
    assert lines[warp[0]] == pl.QILE_ANCHOR
    assert blur[0] == warp[0] + 1 and lines[blur[0]] == BLUR_LINE
    assert pl.QILE_FN.format(sigma=SIGMA) in new_ours
    ks = [n for n in ast.parse(new_ours).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "K1_BLUR_SIGMA"]
    assert len(ks) == 1 and ast.literal_eval(ks[0].value) == 0.65
    a = old_ours.splitlines(keepends=True)
    ops = [op for op in difflib.SequenceMatcher(None, a, lines, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert [op[0] for op in ops] == ["insert", "insert"], ops
    assert "".join(lines[ops[0][3]:ops[0][4]]) == "\n" + pl.QILE_FN.format(sigma=SIGMA) + "\n"
    assert lines[ops[1][3]:ops[1][4]] == [BLUR_LINE]


# ---------------------------------------------------------------- 3: MTH volume blur is off

def _names_used(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


@need_b
def test_run_ours_volume_blur_gated_by_hb(new_ours):
    tree = ast.parse(new_ours)
    assigns = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            assigns.setdefault(n.targets[0].id, []).append(n)
    assert len(assigns["_hb"]) == 1 and len(assigns["_vb"]) == 1
    hb = ast.get_source_segment(new_ours, assigns["_hb"][0].value)
    assert hb == "float(os.environ.get('KNEE_HIRES_BLUR_PX', '0') or 0)", hb
    vb = assigns["_vb"][0].value
    assert isinstance(vb, ast.IfExp) and isinstance(vb.orelse, ast.Constant) and vb.orelse.value is None
    test = vb.test
    assert isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
    gate = test.values[0]
    assert ast.dump(gate) == ast.dump(ast.parse("_hb > 0", mode="eval").body)
    assert "GaussianBlur" in ast.get_source_segment(new_ours, vb.body)
    assert new_ours.count("KNEE_HIRES_BLUR_PX") == 1
    blur_users = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "GaussianBlur"]
    assert len(blur_users) == 2   # _vb (off) and _k1_blur


@need_b
def test_wrapper_passes_repr_hires_and_it_is_zero(built_text):
    tree = ast.parse(built_text)
    stores = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "HIRES_BLUR_PX" and isinstance(n.ctx, ast.Store)]
    assert len(stores) == 1
    kws = [k for k in ast.walk(tree) if isinstance(k, ast.keyword) and k.arg == "KNEE_HIRES_BLUR_PX"]
    assert len(kws) == 1
    assert ast.dump(kws[0].value) == ast.dump(ast.parse("repr(HIRES_BLUR_PX)", mode="eval").body)
    assert built_text.count("KNEE_HIRES_BLUR_PX") == 1
    for k, v in src_values(built_text).items():
        if k != "ours":
            assert "KNEE_HIRES_BLUR_PX" not in bb.unpack(v), k
    hires = ast.literal_eval(hires_node(built_text).value)[1]
    env_value = repr(hires)
    assert env_value == "0.0"
    assert not float(env_value or 0) > 0


# ---------------------------------------------------------------- 4: _k1_blur semantics

@pytest.mark.parametrize("shape", [(5, 224, 224), (2, 31, 47)])
def test_k1_blur_matches_cv2(shape):
    import cv2
    f, s = qile_blur(SIGMA)
    assert s == 0.65
    a = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
    got = f(a)
    assert got.dtype == np.uint8 and got.shape == a.shape and got is not a
    ref = np.stack([np.clip(np.rint(cv2.GaussianBlur(x.astype(np.float32), (0, 0), 0.65,
                                                     borderType=cv2.BORDER_REFLECT_101)), 0, 255).astype(np.uint8) for x in a])
    assert np.array_equal(got, ref)
    assert not np.array_equal(got, a)


def test_k1_blur_sigma0_identity():
    f, s = qile_blur(0)
    assert s == 0.0
    a = np.random.default_rng(8).integers(0, 256, (3, 64, 64), dtype=np.uint8)
    assert f(a) is a


# ---------------------------------------------------------------- 5: metadata

@need_b
def test_metadata():
    old = json.loads(B_META.read_text(encoding="utf-8"))
    new = json.loads((BUILT / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert new["id"] == "sroziews/rsna-kad-bpblur"
    assert new["title"] == "rsna-kad-bpblur"
    assert new["code_file"] == "bpblur.py"
    assert "id_no" not in new
    for k in ("dataset_sources", "docker_image", "machine_shape", "enable_gpu", "enable_internet"):
        assert k in old and new[k] == old[k], k
    assert set(new) == set(old) - {"id_no"}
    assert {k: v for k, v in new.items() if k not in ("id", "title", "code_file")} == \
           {k: v for k, v in old.items() if k not in ("id", "title", "code_file", "id_no")}


# ---------------------------------------------------------------- 6: determinism and guards

@need_b
def test_cli_rebuild_byte_identical(b_text):
    outs = []
    for name in ("r1", "r2"):
        d = SCRATCH / name
        r = run_cli("--src", B_SRC, "--meta", B_META, "--out", d)
        assert r.returncode == 0, r.stderr
        outs.append({p.name: p.read_bytes() for p in sorted(d.iterdir())})
    assert outs[0] == outs[1]
    assert set(outs[0]) == {"bpblur.py", "kernel-metadata.json", "run_ours_patched.py"}
    assert outs[0] == {p.name: p.read_bytes() for p in sorted(BUILT.iterdir()) if p.name in outs[0]}
    assert B_SRC.read_text(encoding="utf-8") == b_text


@pytest.mark.parametrize("sigma", ["0", "-0.5", "nan"])
def test_cli_rejects_nonpositive_sigma(sigma, synthetic):
    d = SCRATCH / f"bad_sigma_{sigma}"
    r = run_cli("--src", synthetic["ok"], "--meta", synthetic["meta"], "--out", d, "--sigma", sigma)
    assert r.returncode != 0
    assert not (d / "bpblur.py").exists()


@pytest.mark.parametrize("sigma", [0, -0.5])
def test_build_rejects_nonpositive_sigma(sigma):
    with pytest.raises(AssertionError):
        bb.build("x = 1\n", sigma)


SYN_OURS = pl.QILE_IMPORT + "def decode(src, M, img, crop_mm):\n    crops = {}\n" + pl.QILE_ANCHOR + "    return crops\n"


def syn_b(src_keys=("hx48", "ours", "r7"), hires=bb.HIRES_OLD):
    srcs = {k: bb.pack(SYN_OURS if k == "ours" else f"print({k!r})\n") for k in src_keys}
    return ("import os\n" + f"SRC = {srcs!r}\n" + "LEGS = 'x'\n" + hires
            + "env = dict(KNEE_HIRES_BLUR_PX=repr(HIRES_BLUR_PX))\n")


@pytest.fixture(scope="module")
def synthetic():
    d = SCRATCH / "syn"
    d.mkdir(parents=True, exist_ok=True)
    files = {"ok": syn_b(), "no_hires": syn_b(hires=""), "hires_changed": syn_b(hires="R7_BLUR_MM, HIRES_BLUR_PX = 0.288, 0.5\n"),
             "hires_twice": syn_b(hires=bb.HIRES_OLD * 2), "no_ours": syn_b(src_keys=("hx48", "r7"))}
    paths = {}
    for k, v in files.items():
        paths[k] = d / f"{k}.py"
        paths[k].write_text(v, encoding="utf-8")
    paths["meta"] = d / "kernel-metadata.json"
    paths["meta"].write_text(json.dumps({"id": "a/b", "id_no": 1, "title": "b", "code_file": "b.py",
                                         "dataset_sources": ["x/y"]}), encoding="utf-8")
    return paths


def test_synthetic_positive_control(synthetic):
    d = SCRATCH / "syn_out_ok"
    r = run_cli("--src", synthetic["ok"], "--meta", synthetic["meta"], "--out", d)
    assert r.returncode == 0, r.stderr
    out = (d / "bpblur.py").read_text(encoding="utf-8")
    assert bb.HIRES_NEW in out and bb.HIRES_OLD not in out
    assert bb.unpack(src_values(out)["ours"]) == pl.patch_qile_src(SYN_OURS, SIGMA)
    assert json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8")) == {
        "id": "sroziews/rsna-kad-bpblur", "title": "rsna-kad-bpblur", "code_file": "bpblur.py", "dataset_sources": ["x/y"]}


@pytest.mark.parametrize("case", ["no_hires", "hires_changed", "hires_twice", "no_ours"])
def test_synthetic_guards_fail_loudly(synthetic, case):
    d = SCRATCH / f"syn_out_{case}"
    r = run_cli("--src", synthetic[case], "--meta", synthetic["meta"], "--out", d)
    assert r.returncode != 0
    assert "Traceback" in r.stderr and ("AssertionError" in r.stderr or "KeyError" in r.stderr), r.stderr[-500:]
    assert not (d / "bpblur.py").exists()


def test_build_without_src_fails():
    with pytest.raises(AssertionError):
        bb.build("x = 1\n" + bb.HIRES_OLD, SIGMA)


# ---------------------------------------------------------------- 7: the qa/blur suites still pass

@pytest.mark.parametrize("name", ["test_patch_legs.py", "test_patch_r7.py"])
def test_blur_suites_pass(name):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE.parent / "blur" / name)],
                       capture_output=True, text=True, cwd=str(HERE.parents[1]))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
