"""Tests for qa/bprefit/build_bprefit.py on synthetic inputs (tmp_path), no real kernel needed:

  PYTHONDONTWRITEBYTECODE=1 python -m pytest qa/bprefit -q -p no:cacheprovider
"""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_bprefit", HERE / "build_bprefit.py")
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)

FINDINGS = ["Baker's cyst", "effusion", "meniscus"]


def stk(seed=0.0, features=None, members=None, note="old"):
    feats = features or ["MTR", "MTH", "OLP", "f\"q\\x"]
    return {
        "features": feats,
        "members": members or ["MTR", "MTH", "OLP"],
        "lr": {f: [0.1 + seed, -0.2 / 3 + seed, 1e-17 + seed, 123456.789012345678] for f in FINDINGS},
        "ranklr": {f: [0.3 + seed, 2.0 / 7] for f in FINDINGS},
        "note": note,
    }


def kernel(old, eol="\n"):
    lines = [
        "import json",
        "# header comment with STK = json.loads( not at line start",
        "X = 1",
        bb.PREFIX + repr(json.dumps(old)) + ")",
        "def f():",
        "    return STK['lr']",
        "",
    ]
    return eol.join(lines)


def new_stk():
    return stk(seed=0.123456789, note="Baker's \"cyst\" \\ path C:\\x\n tab\t ünï")


def test_only_stk_line_changes():
    bp = kernel(stk())
    out, old = bb.build(bp, new_stk())
    a, b = bp.split("\n"), out.split("\n")
    assert len(a) == len(b)
    diff = [k for k in range(len(a)) if a[k] != b[k]]
    assert diff == [3]
    assert old == stk()
    assert out.endswith("\n")


def test_new_line_parses_and_roundtrips_exactly():
    new = new_stk()
    out, _ = bb.build(kernel(stk()), new)
    line = out.split("\n")[3]
    tree = ast.parse(line)
    call = tree.body[0].value
    lit = ast.literal_eval(call.args[0])
    assert json.loads(lit) == new
    # floats bit-exact
    got = json.loads(lit)
    for f in FINDINGS:
        assert [x.hex() for x in got["lr"][f]] == [x.hex() for x in new["lr"][f]]
    # the line actually executes to the same object
    ns = {}
    exec(compile("import json\n" + line, "<stk>", "exec"), ns)
    assert ns["STK"] == new


@pytest.mark.parametrize("note", ["it's", 'say "hi"', "back\\slash", "'\"\\'\"", "\\u0041 not an escape", "}\")"])
def test_quotes_backslashes_roundtrip(note):
    new = stk(seed=1.0, note=note)
    new["features"] = stk()["features"]
    out, _ = bb.build(kernel(stk()), new)
    assert bb.parse_stk(out.split("\n")[3]) == new


def test_refuses_feature_order_change():
    old = stk()
    new = new_stk()
    new["features"] = list(reversed(old["features"]))
    with pytest.raises(AssertionError):
        bb.build(kernel(old), new)


def test_refuses_members_change():
    new = new_stk()
    new["members"] = ["MTR", "OLP", "MTH"]
    with pytest.raises(AssertionError):
        bb.build(kernel(stk()), new)


def test_refuses_extra_feature():
    new = new_stk()
    new["features"] = new["features"] + ["extra"]
    with pytest.raises(AssertionError):
        bb.build(kernel(stk()), new)


@pytest.mark.parametrize("key", ["lr", "ranklr"])
def test_refuses_finding_set_change(key):
    new = new_stk()
    new[key]["new finding"] = new[key][FINDINGS[0]]
    with pytest.raises(AssertionError):
        bb.build(kernel(stk()), new)


def test_refuses_missing_finding_in_lr():
    new = new_stk()
    del new["lr"][FINDINGS[1]]
    with pytest.raises(AssertionError):
        bb.build(kernel(stk()), new)


def test_refuses_zero_stk_lines():
    bp = kernel(stk()).replace(bb.PREFIX, "STK=json.loads(")
    with pytest.raises(AssertionError):
        bb.build(bp, new_stk())


def test_refuses_two_stk_lines():
    old = stk()
    bp = kernel(old) + bb.PREFIX + repr(json.dumps(old)) + ")\n"
    with pytest.raises(AssertionError):
        bb.build(bp, new_stk())


def test_indented_stk_line_not_counted():
    bp = kernel(stk()) + "    " + bb.PREFIX + "'{}')\n"
    out, _ = bb.build(bp, new_stk())
    assert out.split("\n")[-2] == "    " + bb.PREFIX + "'{}')"


def run_main(tmp_path, bp_bytes, meta):
    src = tmp_path / "in"
    src.mkdir()
    (src / "bpblur.py").write_bytes(bp_bytes)
    (src / "stk.json").write_text(json.dumps(new_stk()), encoding="utf-8")
    (src / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "out" / "nested"
    argv = ["build_bprefit.py", "--bpblur", str(src / "bpblur.py"), "--stk", str(src / "stk.json"),
            "--meta", str(src / "kernel-metadata.json"), "--out", str(out)]
    old, sys.argv = sys.argv, argv
    try:
        bb.main()
    finally:
        sys.argv = old
    return out


META = {
    "id": "sroziews/rsna-kad-bpblur",
    "id_no": 123456,
    "title": "rsna-kad-bpblur",
    "code_file": "bpblur.py",
    "language": "python",
    "kernel_type": "script",
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": ["a/b"],
    "competition_sources": ["rsna-2026"],
    "kernel_sources": ["x/y"],
    "model_sources": ["m/n/o/p"],
}


def test_main_writes_kernel_and_metadata(tmp_path, capsys):
    bp = kernel(stk())
    out = run_main(tmp_path, bp.encode("utf-8"), META)
    code = (out / "bprefit.py").read_bytes()
    assert b"\r\n" not in code
    expected, _ = bb.build(bp, new_stk())
    assert code == expected.encode("utf-8")
    raw = (out / "kernel-metadata.json").read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    meta = json.loads(raw)
    assert meta["id"] == "sroziews/rsna-kad-bprefit"
    assert meta["title"] == "rsna-kad-bprefit"
    assert meta["code_file"] == "bprefit.py"
    assert "id_no" not in meta
    for k in META:
        if k not in ("id", "id_no", "title", "code_file"):
            assert meta[k] == META[k], k
    assert "max |d lr| vs old" in capsys.readouterr().out


def test_main_without_id_no(tmp_path):
    meta = {k: v for k, v in META.items() if k != "id_no"}
    out = run_main(tmp_path, kernel(stk()).encode("utf-8"), meta)
    assert json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))["id"] == bb.KERNEL_ID


def test_main_non_ascii_source_kept(tmp_path):
    bp = kernel(stk()).replace("X = 1", "X = 'żółć µ'")
    out = run_main(tmp_path, bp.encode("utf-8"), META)
    code = (out / "bprefit.py").read_bytes().decode("utf-8")
    assert code.split("\n")[2] == "X = 'żółć µ'"
