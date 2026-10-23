"""Tests for qa/bblur/build_bpsyn.py (run on the server, CPU only):

  CUDA_VISIBLE_DEVICES= python -I -m pytest -q -p no:cacheprovider qa/bblur/test_build_bpsyn.py

Andrey's flat4-syn-r6 kernel (FLAT4, env override) is untrusted data: it is only read and parsed with ast, never run.
The only code exec'd is the final-blend tail of our own bpblur.py / bpsyn.py, on synthetic member predictions.
The real-input tests skip when the inputs are absent."""
import ast
import difflib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_bpsyn.py"
spec = importlib.util.spec_from_file_location("build_bpsyn", BUILD)
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
BPBLUR = Path(os.environ.get("BPBLUR", HERE / "kernel_bpblur" / "bpblur.py"))
BPBLUR_META = Path(os.environ.get("BPBLUR_META", HERE / "kernel_bpblur" / "kernel-metadata.json"))
FLAT4 = Path(os.environ.get("FLAT4", ROOT / ".tmp/research_andrey/flat4synr6/rsna-knee-flat4-syn-r6.py"))
BUILT = Path(os.environ.get("BPSYN_BUILT", HERE / "kernel_bpsyn"))
SCRATCH = Path(os.environ.get("BPSYN_TEST_SCRATCH", ROOT / ".tmp/test_bpsyn"))
MD5_BPBLUR = "32cf28a992b16166a433154f00b0b43e"
MD5_BPSYN = "d045e2495638c7294009ec81034678c5"
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
          'Synovitis', "Baker's", 'Contusion', 'Fracture']
MEMBERS = ("r7", "MTR", "MTH", "OLP", "v5c", "hx48")
TAIL_START, TAIL_END = "n = len(ids)\n", "sub.to_csv("

need_real = pytest.mark.skipif(not (BPBLUR.exists() and FLAT4.exists()), reason=f"{BPBLUR} or {FLAT4} missing")


def md5(b):
    return hashlib.md5(b).hexdigest()


def assign_index(body, name):
    idx = [i for i, n in enumerate(body) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name]
    assert len(idx) == 1, (name, len(idx))
    return idx[0]


def assign(text, name):
    body = ast.parse(text).body
    return body[assign_index(body, name)]


def json_literal(text, name):
    """(source text of the string literal in `name = json.loads(<lit>)`, parsed JSON) -- written independently of bs."""
    call = assign(text, name).value
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == "json.loads" and len(call.args) == 1
    lit = call.args[0]
    assert isinstance(lit, ast.Constant) and isinstance(lit.value, str)
    return ast.get_source_segment(text, lit), json.loads(lit.value)


def tail(text):
    assert text.count(TAIL_START) == 1 and text.count(TAIL_END) == 1
    i = text.index(TAIL_START)
    return text[i:text.index(TAIL_END, i)]


def run_cli(*args):
    return subprocess.run([sys.executable, "-I", str(BUILD), *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def bp():
    return BPBLUR.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def flat4():
    return FLAT4.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def built():
    return (BUILT / "bpsyn.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def stk(flat4):
    return json_literal(flat4, "STK")


# ---------------------------------------------------------------- 1: SYN == flat4 STK

@need_real
def test_syn_equals_flat4_stk(built, stk):
    lit, syn = json_literal(built, "SYN")
    assert syn == stk[1]
    assert lit == stk[0]
    assert lit.encode("utf-8") in FLAT4.read_bytes()
    assert syn["finding"] == "Synovitis" and syn["share"] == 0.25
    assert len(syn["features"]) == 36 == len(syn["mean"]) == len(syn["std"]) == len(syn["lr"]) == len(syn["ranklr"])
    assert {f.split(" ", 1)[0] for f in syn["features"]} == {"MT-hires", "OL-plane", "hx48"}
    assert {f.split(" ", 1)[1] for f in syn["features"]} == set(LABELS)


# ---------------------------------------------------------------- 2: only the block is inserted

@need_real
def test_built_equals_fresh_build(bp, flat4, built):
    out, syn = bs.build(bp, flat4)
    assert out == built
    assert md5(bp.encode("utf-8")) == MD5_BPBLUR


@need_real
def test_only_block_inserted_before_anchor(bp, built, stk):
    assert bp.count(bs.ANCHOR) == 1 and built.count(bs.ANCHOR) == 1
    a, b = bp.splitlines(keepends=True), built.splitlines(keepends=True)
    ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert [op[0] for op in ops] == ["insert"], ops
    _, i1, i2, j1, j2 = ops[0]
    assert a[i1] == bs.ANCHOR and b[j2] == bs.ANCHOR
    block = "".join(b[j1:j2])
    assert block == bs.BLOCK.replace("@@LIT@@", stk[0])
    assert block.startswith("# [bpsyn] ") and "@@LIT@@" not in built
    assert a[:i1] == b[:j1] and a[i1:] == b[j2:]
    compile(built, "bpsyn.py", "exec")


def _renamed(node, names):
    class R(ast.NodeTransformer):
        def visit_Name(self, n):
            return ast.copy_location(ast.Name(id=names.get(n.id, n.id), ctx=n.ctx), n)
    return ast.dump(R().visit(node))


@need_real
def test_block_statements_are_flat4_statements_renamed(built, flat4):
    """The block's code (all but the JSON line and the log line) is flat4's Synovitis stacker with names renamed."""
    names = {"STK": "SYN", "KEY": "KEY_SYN", "X": "XS", "stack": "stack_syn", "j": "j_syn"}
    f_body = ast.parse(flat4).body
    i = assign_index(f_body, "STK")
    f_stmts = [s for s in f_body[i + 1:] if not (isinstance(s, ast.FunctionDef) or
                                                (isinstance(s, ast.Assign) and getattr(s.targets[0], "id", "") == "RK"))]
    f_stmts = f_stmts[:6]   # KEY, X, X-norm, stack, j, out[:, j]
    b_body = ast.parse(built).body
    k = assign_index(b_body, "SYN")
    b_stmts = b_body[k + 1:k + 7]
    assert [_renamed(s, names) for s in f_stmts] == [ast.dump(s) for s in b_stmts]
    say = b_body[k + 7]
    assert ast.unparse(say).startswith("say('[bpsyn] Synovitis stacker")


# ---------------------------------------------------------------- 3: the final blend on synthetic members

def ref_prk(v):
    """Percentile rank, ties averaged, over finite values; non-finite -> 0.5 (numpy only, independent of pandas)."""
    v = np.asarray(v, float)
    r = np.full(v.shape, 0.5)
    ok = np.isfinite(v)
    x = v[ok]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    s = 0
    while s < len(xs):
        e = s
        while e + 1 < len(xs) and xs[e + 1] == xs[s]:
            e += 1
        ranks[order[s:e + 1]] = (s + e) / 2 + 1
        s = e + 1
    r[ok] = ranks / len(x)
    return r


def synthetic_P(seed, n):
    rng = np.random.default_rng(seed)
    ids = [f"1.2.3.{seed}.{i}" for i in range(n)]
    P = {}
    for k in MEMBERS:
        a = rng.random((n, len(LABELS)))
        a[rng.random(a.shape) < 0.1] = np.nan                  # a member missing a study
        a[:, 3] = np.round(a[:, 3], 1)                         # ties
        P[k] = pd.DataFrame(a, index=ids, columns=LABELS)
    return ids, P


def run_tail(text, ids, P):
    ns = {"np": np, "pd": pd, "json": json, "labels": list(LABELS), "ids": list(ids), "P": P, "ID": "StudyInstanceUID",
          "say": lambda *m: None}
    exec(compile(tail(text), "<tail>", "exec"), ns)
    return ns


def ref_synovitis(base, P, stk, key):
    rk = {k: {c: ref_prk(P[k][c].values) for c in LABELS} for k in P}
    xs = np.stack([rk[key[f.split(" ", 1)[0]]][f.split(" ", 1)[1]] for f in stk["features"]], 1)
    xs = (xs - np.array(stk["mean"])) / np.array(stk["std"])
    stack = 0.5 * ref_prk(xs @ np.array(stk["lr"])) + 0.5 * ref_prk(xs @ np.array(stk["ranklr"]))
    return 0.75 * ref_prk(base) + 0.25 * ref_prk(stack)


def test_ref_prk_matches_pandas_rank():
    v = np.array([3., np.nan, 1., 3., 2., np.inf * 0, 5.])
    want = pd.Series(v).rank(method="average", pct=True).values
    assert np.allclose(ref_prk(v), np.where(np.isfinite(want), want, 0.5), atol=0, rtol=0)


@need_real
def test_flat4_formula_lines(flat4):
    """Pin the flat4 lines the reference recomputation is written from."""
    body = {ast.unparse(s.targets[0]): ast.unparse(s.value) for s in ast.parse(flat4).body if isinstance(s, ast.Assign)}
    assert body["stack"] == "0.5 * prk(X @ np.array(STK['lr'])) + 0.5 * prk(X @ np.array(STK['ranklr']))"
    assert body["out[:, j]"] == "(1 - STK['share']) * prk(out[:, j]) + STK['share'] * prk(stack)"
    assert body["X"] == "(X - np.array(STK['mean'])) / np.array(STK['std'])"
    assert ast.literal_eval(assign(flat4, "KEY").value) == {"MT-hires": "MTH", "OL-plane": "OLP", "hx48": "hx48"}


@need_real
@pytest.mark.parametrize("seed,n", [(0, 300), (1, 57), (2, 1000)])
def test_tail_blend(bp, built, flat4, stk, seed, n):
    ids, P = synthetic_P(seed, n)
    old, new = run_tail(bp, ids, P), run_tail(built, ids, P)
    j = LABELS.index("Synovitis")
    others = [c for c in LABELS if c != "Synovitis"]
    assert new["out"].shape == old["out"].shape == (n, 12)
    assert np.array_equal(np.delete(new["out"], j, 1), np.delete(old["out"], j, 1))
    assert new["sub"][others].equals(old["sub"][others])
    assert list(new["sub"].columns) == list(old["sub"].columns) == ["StudyInstanceUID"] + LABELS
    assert list(new["sub"]["StudyInstanceUID"]) == ids
    key = ast.literal_eval(assign(flat4, "KEY").value)
    want = ref_synovitis(old["out"][:, j], P, stk[1], key)
    assert np.allclose(new["out"][:, j], want, rtol=0, atol=1e-12)
    assert np.array_equal(new["sub"]["Synovitis"].values, want.astype(np.float32))
    assert not np.allclose(new["out"][:, j], old["out"][:, j], atol=1e-3)   # the stacker does change Synovitis
    assert new["XS"].shape == (n, 36)


# ---------------------------------------------------------------- 4: guards

def _flat4_with(flat4, stk, **change):
    lit, d = stk
    d = dict(d, **change)
    new = flat4.replace(lit, repr(json.dumps(d)))
    assert (new != flat4) == bool(change) and flat4.count(lit) == 1
    return new


@need_real
def test_build_rejects_missing_anchor(bp, flat4):
    with pytest.raises(AssertionError):
        bs.build(bp.replace(bs.ANCHOR, "say('other')\n"), flat4)


@need_real
def test_build_rejects_duplicated_anchor(bp, flat4):
    with pytest.raises(AssertionError):
        bs.build(bp.replace(bs.ANCHOR, bs.ANCHOR * 2), flat4)
    with pytest.raises(AssertionError):
        bs.build(bp + bs.ANCHOR, flat4)


@need_real
@pytest.mark.parametrize("change", [{"finding": "Effusion"}, {"share": 0.3}])
def test_build_rejects_other_stacker(bp, flat4, stk, change):
    with pytest.raises(AssertionError):
        bs.build(bp, _flat4_with(flat4, stk, **change))


@need_real
def test_build_positive_control_on_modified_flat4(bp, flat4, stk):
    """The same rewrite with an unchanged dict still builds (the guard tests above fail for the right reason)."""
    out, syn = bs.build(bp, _flat4_with(flat4, stk))
    assert syn == stk[1]


def test_build_rejects_flat4_without_stk():
    with pytest.raises(AssertionError):
        bs.build(bs.ANCHOR, "x = 1\n")


@need_real
@pytest.mark.parametrize("case", ["no_anchor", "two_anchors", "effusion"])
def test_cli_guards_fail_loudly(bp, flat4, stk, case):
    d = SCRATCH / f"bad_{case}"
    d.mkdir(parents=True, exist_ok=True)
    b, f = bp, flat4
    if case == "no_anchor":
        b = bp.replace(bs.ANCHOR, "")
    elif case == "two_anchors":
        b = bp.replace(bs.ANCHOR, bs.ANCHOR * 2)
    else:
        f = _flat4_with(flat4, stk, finding="Effusion")
    (d / "bp.py").write_text(b, encoding="utf-8")
    (d / "f4.py").write_text(f, encoding="utf-8")
    out = d / "out"
    r = run_cli("--bpblur", d / "bp.py", "--flat4", d / "f4.py", "--meta", BPBLUR_META, "--out", out)
    assert r.returncode != 0 and "AssertionError" in r.stderr, r.stderr[-500:]
    assert not (out / "bpsyn.py").exists()


# ---------------------------------------------------------------- 5: real build reproduces the server output

@need_real
def test_cli_real_build_md5(bp):
    flat4_before = FLAT4.read_bytes()
    outs = []
    for name in ("r1", "r2"):
        d = SCRATCH / name
        r = run_cli("--bpblur", BPBLUR, "--flat4", FLAT4, "--meta", BPBLUR_META, "--out", d)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == f"bpblur {MD5_BPBLUR} -> bpsyn {MD5_BPSYN}", r.stdout
        outs.append({p.name: p.read_bytes() for p in sorted(d.iterdir())})
    assert outs[0] == outs[1]
    assert set(outs[0]) == {"bpsyn.py", "kernel-metadata.json"}
    assert md5(outs[0]["bpsyn.py"]) == MD5_BPSYN
    assert md5(BPBLUR.read_bytes()) == MD5_BPBLUR
    assert outs[0] == {p.name: p.read_bytes() for p in sorted(BUILT.iterdir()) if p.name in outs[0]}
    assert FLAT4.read_bytes() == flat4_before


@need_real
def test_metadata():
    old = json.loads(BPBLUR_META.read_text(encoding="utf-8"))
    new = json.loads((BUILT / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert new["id"] == "sroziews/rsna-kad-bpsyn" and new["title"] == "rsna-kad-bpsyn" and new["code_file"] == "bpsyn.py"
    assert "id_no" not in new
    assert {k: v for k, v in new.items() if k not in ("id", "title", "code_file")} == \
           {k: v for k, v in old.items() if k not in ("id", "title", "code_file", "id_no")}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
