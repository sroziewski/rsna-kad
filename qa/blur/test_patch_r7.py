"""Tests for qa/blur/patch_r7.py (run on the server, CPU only):

  CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider qa/blur/test_patch_r7.py

The real-notebook tests read R7_NB (env override) and skip when it is absent. The blur code is taken out of the patched
probe_infer.py text (only K1_BLUR_MM, _k1_blur_install and install_native are exec'd), never by running heliosli's code."""
import ast
import dataclasses
import difflib
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("patch_r7", HERE / "patch_r7.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
R7_NB = Path(os.environ.get("R7_NB", ROOT / "qa/v3stack/src/0-957-latest-dataset-score-in-one-click.ipynb"))
GEOM_PY = Path(os.environ.get("R7_GEOM", ROOT / ".tmp/s6r/S6F0/heliosli/cloud_code/training/dicom_geometry.py"))
SCRATCH = Path(os.environ.get("BLUR_TEST_SCRATCH", ROOT / ".tmp/blur/test_r7"))
SIGMA = 0.288
need_nb = pytest.mark.skipif(not R7_NB.exists(), reason=f"{R7_NB} missing")


def dicts(cell):
    out = {}
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("SOURCES", "SOURCE_SHA"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(out) == {"SOURCES", "SOURCE_SHA"}
    return out["SOURCES"], out["SOURCE_SHA"]


def hit_index(nb):
    code = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    hits = [i for i in code if "SOURCES = {" in "".join(nb["cells"][i]["source"])]
    assert len(hits) == 1
    return hits[0], code.index(hits[0])


def run_patch(out, sigma=SIGMA, nb=R7_NB):
    r = subprocess.run([sys.executable, str(HERE / "patch_r7.py"), "--sigma-mm", str(sigma), "--nb", str(nb),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture(scope="module")
def patched():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    a = run_patch(SCRATCH / "r7_blur_a.ipynb")
    b = run_patch(SCRATCH / "r7_blur_b.ipynb")
    orig = json.loads(R7_NB.read_text(encoding="utf-8"))
    new = json.loads(a.read_text(encoding="utf-8"))
    return SimpleNamespace(a=a, b=b, orig=orig, new=new)


@pytest.fixture(scope="module")
def texts(patched):
    i, _ = hit_index(patched.orig)
    oc, nc = "".join(patched.orig["cells"][i]["source"]), "".join(patched.new["cells"][i]["source"])
    return SimpleNamespace(i=i, oc=oc, nc=nc, od=dicts(oc), nd=dicts(nc))


# ---------- 1. notebook-level diff ----------

@need_nb
def test_only_code_cell_2_differs(patched):
    i, code_pos = hit_index(patched.orig)
    assert code_pos == 2
    assert {k: v for k, v in patched.orig.items() if k != "cells"} == {k: v for k, v in patched.new.items() if k != "cells"}
    assert len(patched.orig["cells"]) == len(patched.new["cells"])
    differing = [j for j, (o, n) in enumerate(zip(patched.orig["cells"], patched.new["cells"])) if o != n]
    assert differing == [i]
    o, n = patched.orig["cells"][i], patched.new["cells"][i]
    assert {k: v for k, v in o.items() if k != "source"} == {k: v for k, v in n.items() if k != "source"}


@need_nb
def test_only_probe_infer_entries_change(texts):
    (os_, osha), (ns, nsha) = texts.od, texts.nd
    assert list(os_) == list(ns) and list(osha) == list(nsha)
    assert [k for k in os_ if os_[k] != ns[k]] == ["probe_infer.py"]
    assert [k for k in osha if osha[k] != nsha[k]] == ["probe_infer.py"]


@need_nb
def test_new_sha_matches_text(texts):
    for src, sha in (texts.od, texts.nd):
        for k in src:
            assert hashlib.sha256(src[k].encode("utf-8")).hexdigest() == sha[k], k


@need_nb
def test_cell_text_changes_only_in_the_two_literals(texts):
    """Outside the two value literals, the cell text is byte-identical."""
    def spans(cell):
        out = []
        tree = ast.parse(cell)
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("SOURCES", "SOURCE_SHA"):
                v = dict(zip([k.value for k in node.value.keys], node.value.values))["probe_infer.py"]
                out.append(v)
        lines = cell.encode("utf-8").splitlines(keepends=True)
        off = lambda ln, col: sum(len(x) for x in lines[:ln - 1]) + col
        return [(off(v.lineno, v.col_offset), off(v.end_lineno, v.end_col_offset)) for v in out]
    ob, nb = texts.oc.encode("utf-8"), texts.nc.encode("utf-8")
    (o1s, o1e), (o2s, o2e) = spans(texts.oc)
    (n1s, n1e), (n2s, n2e) = spans(texts.nc)
    assert ob[:o1s] == nb[:n1s]
    assert ob[o1e:o2s] == nb[n1e:n2s]
    assert ob[o2e:] == nb[n2e:]


@need_nb
def test_probe_infer_diff_is_block_plus_call(texts):
    old, new = texts.od[0]["probe_infer.py"], texts.nd[0]["probe_infer.py"]
    ol, nl = old.splitlines(keepends=True), new.splitlines(keepends=True)
    ops = [op for op in difflib.SequenceMatcher(None, ol, nl, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert [op[0] for op in ops] == ["insert", "insert"]
    block = "".join(nl[ops[0][3]:ops[0][4]])
    call = "".join(nl[ops[1][3]:ops[1][4]])
    assert ol[ops[0][1] - 1] == pr.ANCHOR_TOP
    assert ol[ops[1][1] - 1] == pr.ANCHOR_CALL
    assert block == "\n" + pr.FN.format(sigma=SIGMA)
    assert call == "    _k1_blur_install(native)   # [blur probe]\n"
    assert f"K1_BLUR_MM = {SIGMA!r}   #" in block


# ---------- 2. blur behaviour ----------

@dataclasses.dataclass(frozen=True)
class Payload:
    image: np.ndarray
    spacing: tuple
    row_axis: np.ndarray
    col_axis: np.ndarray
    photometric_interpretation: str = "MONOCHROME2"
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0


def blur_ns(probe_text, sigma):
    """exec K1_BLUR_MM + _k1_blur_install out of probe_infer.py, with K1_BLUR_MM forced to sigma."""
    tree = ast.parse(probe_text)
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "K1_BLUR_MM")
            or (isinstance(n, ast.FunctionDef) and n.name == "_k1_blur_install")]
    assert len(keep) == 2
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "probe_infer_extract", "exec"), ns)
    ns["K1_BLUR_MM"] = sigma
    return ns


def fake_native(payload):
    calls = []

    def pixels_from_dataset(ds, image=None, path="archive"):
        calls.append((ds, image, path))
        return payload
    return SimpleNamespace(geom=SimpleNamespace(pixels_from_dataset=pixels_from_dataset)), calls


def payload(image, spacing=(0.3, 0.6)):
    return Payload(image=image, spacing=spacing, row_axis=np.array([0.0, 1.0, 0.0]), col_axis=np.array([0.0, 0.0, -1.0]),
                   photometric_interpretation="MONOCHROME1", rescale_slope=2.5, rescale_intercept=-1024.0)


def ref_blur(img2d, mm, spacing, swap=False):
    sy, sx = mm / spacing[0], mm / spacing[1]
    if swap:
        sy, sx = sx, sy
    return cv2.GaussianBlur(np.asarray(img2d, np.float32), (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_REFLECT_101)


@pytest.fixture(scope="module")
def probe_text():
    if R7_NB.exists():
        nb = json.loads(R7_NB.read_text(encoding="utf-8"))
        i, _ = hit_index(nb)
        src = dicts("".join(nb["cells"][i]["source"]))[0]["probe_infer.py"]
    else:   # synthetic stand-in carrying both anchors, so the blur tests still run without the notebook
        src = ('import numpy as np\n' + pr.ANCHOR_TOP + '\n\ndef install_native(lean=True):\n' + pr.ANCHOR_CALL +
               '    return native  # pixels_from_dataset pixels_from_dataset\n')
    return pr.patch_probe_infer(src, SIGMA)


def test_sigma_zero_is_noop(probe_text):
    ns = blur_ns(probe_text, 0.0)
    native, _ = fake_native(payload(np.zeros((8, 8), np.float32)))
    fn = native.geom.pixels_from_dataset
    ns["_k1_blur_install"](native)
    assert native.geom.pixels_from_dataset is fn
    assert not hasattr(native.geom, "_k1_blur")


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.float64])
def test_blur_matches_cv2_anisotropic(probe_text, dtype):
    rng = np.random.default_rng(0)
    img = (rng.random((37, 53)) * 4000).astype(dtype)   # non-square H != W
    spacing = (0.3, 0.6)
    ns = blur_ns(probe_text, SIGMA)
    native, _ = fake_native(payload(img, spacing))
    ns["_k1_blur_install"](native)
    out = native.geom.pixels_from_dataset("ds").image
    want = ref_blur(img, SIGMA, spacing)
    assert out.dtype == np.float32 and out.shape == img.shape
    np.testing.assert_array_equal(out, want)
    swapped = ref_blur(img, SIGMA, spacing, swap=True)
    assert np.abs(out - swapped).max() > 1.0
    assert np.abs(out - np.asarray(img, np.float32)).max() > 1.0


def test_blur_3d_per_frame(probe_text):
    rng = np.random.default_rng(1)
    img = (rng.random((4, 29, 41)) * 1000).astype(np.float32)
    img[2] = 0.0
    img[3, 10, 10] = 5000.0
    spacing = (0.25, 0.5)
    ns = blur_ns(probe_text, SIGMA)
    native, _ = fake_native(payload(img, spacing))
    ns["_k1_blur_install"](native)
    out = native.geom.pixels_from_dataset("ds").image
    assert out.shape == img.shape
    for f in range(img.shape[0]):
        np.testing.assert_array_equal(out[f], ref_blur(img[f], SIGMA, spacing))
    assert not out[2].any()   # no mixing across frames


def test_fields_preserved_and_input_untouched(probe_text):
    img = np.arange(30 * 20, dtype=np.float32).reshape(30, 20) % 97
    keep = img.copy()
    p = payload(img)
    ns = blur_ns(probe_text, SIGMA)
    native, calls = fake_native(p)
    ns["_k1_blur_install"](native)
    sentinel = object()
    q = native.geom.pixels_from_dataset("DS", image=sentinel, path="a/b.dcm")
    assert calls == [("DS", sentinel, "a/b.dcm")]
    native.geom.pixels_from_dataset("DS2", path="x")
    assert calls[-1] == ("DS2", None, "x")
    assert type(q) is Payload
    for f in dataclasses.fields(Payload):
        if f.name != "image":
            assert getattr(q, f.name) is getattr(p, f.name), f.name
    np.testing.assert_array_equal(img, keep)
    np.testing.assert_array_equal(p.image, keep)
    assert q.image is not img


def test_install_twice_wraps_once(probe_text):
    img = np.random.default_rng(2).random((16, 24)).astype(np.float32) * 100
    ns = blur_ns(probe_text, SIGMA)
    native, calls = fake_native(payload(img))
    ns["_k1_blur_install"](native)
    first = native.geom.pixels_from_dataset
    ns["_k1_blur_install"](native)
    assert native.geom.pixels_from_dataset is first
    out = native.geom.pixels_from_dataset("ds").image
    assert len(calls) == 1
    np.testing.assert_array_equal(out, ref_blur(img, SIGMA, (0.3, 0.6)))


@pytest.mark.skipif(not GEOM_PY.exists(), reason=f"{GEOM_PY} missing")
def test_wrapper_signature_matches_real_decoder(probe_text):
    real = [n for n in ast.parse(GEOM_PY.read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef) and n.name == "pixels_from_dataset"]
    assert len(real) == 1
    ns = blur_ns(probe_text, SIGMA)
    native, _ = fake_native(payload(np.zeros((4, 4), np.float32)))
    ns["_k1_blur_install"](native)
    got = inspect.signature(native.geom.pixels_from_dataset)
    a = real[0].args
    assert [p.name for p in got.parameters.values()] == [x.arg for x in a.args]
    assert [p.default for p in got.parameters.values() if p.default is not inspect.Parameter.empty] == \
        [ast.literal_eval(d) for d in a.defaults]


@need_nb
def test_call_placed_right_after_native_import(texts):
    old, new = texts.od[0]["probe_infer.py"], texts.nd[0]["probe_infer.py"]

    def fn_src(text):
        node = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "install_native"]
        assert len(node) == 1
        return ast.get_source_segment(text, node[0]) + "\n"
    o, n = fn_src(old), fn_src(new)
    assert pr.ANCHOR_CALL in o and o.count(pr.ANCHOR_CALL) == 1
    assert pr.ANCHOR_CALL + "    _k1_blur_install(native)   # [blur probe]\n" in n
    assert n.replace("    _k1_blur_install(native)   # [blur probe]\n", "", 1) == o
    assert "_k1_blur_install" not in o
    # the original edits are still in place
    for s in ("inspect.getsource(native.prepare)", "stop_before_pixels=True", "native.quantize_series = ",
              "native.DirectorySource = OneStudySource"):
        assert s in n


FAKE_PREPARE = '''import io


def prepare(data, work, split=None, study_ids=None, source_directory=False):
    a = pydicom.dcmread(io.BytesIO(archive.read(i)))
    b = geom.pixels_from_dataset(objects[sid][name],path=name)
    return "stock"
'''


@need_nb
@pytest.mark.parametrize("lean", [True, False])
def test_patched_install_native_runs(texts, lean):
    """Run the patched install_native against a fake training.native_cache: its own source edits still apply and the
    decoder ends up wrapped."""
    new = texts.nd[0]["probe_infer.py"]
    tree = ast.parse(new)
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "K1_BLUR_MM")
            or (isinstance(n, ast.FunctionDef) and n.name in ("_k1_blur_install", "install_native"))]
    assert len(keep) == 3
    pkg = SCRATCH / f"fakepkg_{int(lean)}"
    (pkg / "training").mkdir(parents=True, exist_ok=True)
    (pkg / "training" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "training" / "native_cache.py").write_text(
        "from types import SimpleNamespace\n" + FAKE_PREPARE +
        "\n\ndef _pfd(ds, image=None, path='archive'):\n    return ds\n\n\ngeom = SimpleNamespace(pixels_from_dataset=_pfd)\n"
        "quantize_series = 'stock'\nDirectorySource = 'stock'\n", encoding="utf-8")
    for m in [m for m in sys.modules if m == "training" or m.startswith("training.")]:
        del sys.modules[m]
    sys.path.insert(0, str(pkg))
    try:
        ns = {"np": np, "inspect": inspect, "quantize_series_slicewise": "QS", "OneStudySource": "OSS"}
        exec(compile(ast.Module(body=keep, type_ignores=[]), "probe_infer_extract", "exec"), ns)
        native = ns["install_native"](lean)
        assert native.geom._k1_blur is True
        assert native.geom.pixels_from_dataset.__name__ == "blurred"
        assert native.DirectorySource == "OSS"
        if lean:
            assert native.quantize_series == "QS"
            assert native.prepare.__code__.co_filename == "probe_infer:prepare"
        else:
            assert native.quantize_series == "stock"
        img = np.random.default_rng(3).random((12, 18)).astype(np.float32)
        out = native.geom.pixels_from_dataset(payload(img), path="p")   # the fake decoder returns its ds
        np.testing.assert_array_equal(out.image, ref_blur(img, SIGMA, (0.3, 0.6)))
    finally:
        sys.path.remove(str(pkg))
        for m in [m for m in sys.modules if m == "training" or m.startswith("training.")]:
            del sys.modules[m]


# ---------- 3. compile + determinism ----------

@need_nb
def test_patched_probe_compiles(texts):
    compile(texts.nd[0]["probe_infer.py"], "probe_infer.py", "exec")


@need_nb
def test_patched_cell_compiles(texts):
    ast.parse(texts.nc)


@need_nb
def test_two_runs_byte_identical(patched):
    assert patched.a.read_bytes() == patched.b.read_bytes()


@need_nb
def test_sigma_zero_patch_writes_zero(texts):
    out = run_patch(SCRATCH / "r7_blur_s0.ipynb", sigma=0)
    nb = json.loads(out.read_text(encoding="utf-8"))
    src, sha = dicts("".join(nb["cells"][texts.i]["source"]))
    assert "K1_BLUR_MM = 0.0   #" in src["probe_infer.py"]
    assert hashlib.sha256(src["probe_infer.py"].encode("utf-8")).hexdigest() == sha["probe_infer.py"]


# ---------- 4. anchors ----------

BASE = ('import numpy as np\n' + pr.ANCHOR_TOP + '\n\ndef install_native(lean=True):\n' + pr.ANCHOR_CALL +
        '    native.geom.pixels_from_dataset\n    return native  # pixels_from_dataset\n')


def test_synthetic_base_patches():
    pr.patch_probe_infer(BASE, SIGMA)


@pytest.mark.parametrize("bad", [
    BASE.replace(pr.ANCHOR_TOP, 'UIDK = "SeriesInstanceUID"\n'),
    BASE.replace(pr.ANCHOR_CALL, "    import training.native_cache as nc\n"),
    BASE + pr.ANCHOR_TOP,
    BASE + "\ndef f():\n" + pr.ANCHOR_CALL,
    BASE.replace("return native  # pixels_from_dataset", "return native"),
    BASE + "# pixels_from_dataset\n",
], ids=["no_top", "no_call", "dup_top", "dup_call", "hook_count_1", "hook_count_3"])
def test_missing_or_ambiguous_anchor_raises(bad):
    with pytest.raises(AssertionError):
        pr.patch_probe_infer(bad, SIGMA)


def test_missing_source_sha_dict_raises():
    cell = f"SOURCES = {{'probe_infer.py': {BASE!r}}}\n"
    with pytest.raises(AssertionError):
        pr.patch_cell(cell, SIGMA)


@need_nb
def test_real_cell_with_anchor_removed_raises(texts):
    broken = pr.replace_literal(texts.oc, pr.dict_value_pick(texts.oc, "SOURCES", "probe_infer.py"),
                                lambda v: v.replace(pr.ANCHOR_CALL, "    import training.native_cache as nc\n"))
    assert broken != texts.oc
    with pytest.raises(AssertionError):
        pr.patch_cell(broken, SIGMA)
