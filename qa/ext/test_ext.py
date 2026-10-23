"""Synthetic tests for qa/ext/convert_ext.py and qa/ext/make_inputs_ext.py (tester, 2026-10-16).

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python qa/ext/test_ext.py [--keep] [--pck-format py2|p2|p3|p4]

The synthetic KneeMRI .pck files default to the archive's own format: Python 3 pickle protocol 2, whose array
bytes go through _codecs.encode(..., 'latin1').

Everything is written under .tmp/ext_test/synth_<pid>; exits 1 when any check fails.
"""
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(HERE)]
import convert_ext as ce  # noqa: E402
import make_inputs_ext as mi  # noqa: E402
from rsna_kad.cache import INDEX_COLUMNS, CacheIndex, _valid_npy, load_study, series_path  # noqa: E402
from rsna_kad.config import load_cache_config, params_sha256  # noqa: E402
from rsna_kad.constants import ID_COL, LABEL_KEYS, LABELS, MAIN_SLOTS, SERIES_COL, SLOTS  # noqa: E402
from rsna_kad.dicom import Geometry  # noqa: E402
from rsna_kad.preprocess import series_to_volume  # noqa: E402

CFG_PATH = ROOT / "configs/cache_v1.yaml"
CFG = load_cache_config(CFG_PATH)
SHA = params_sha256(CFG)
REAL_V1_MANIFEST = Path("/raid/sroziewski/cache/rsna_kad/v1/manifest.json")
PY = sys.executable
CONVERT = str(HERE / "convert_ext.py")

RESULTS = []


def check(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                fn(*a, **k)
                RESULTS.append((name, True, ""))
                print(f"PASS {name} ({time.time() - t0:.1f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        return run
    return deco


# ------------------------------------------------------------------ synthetic sources

def phantom(n, size, dtype, seed):
    """Knee-ish phantom: a bright off-centre disc on a ramp plus noise, so robust windowing is non-trivial."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    top = 4000.0 if dtype == np.uint16 else 250.0
    out = np.empty((n, size, size), dtype)
    for i in range(n):
        disc = ((yy - size * 0.45) ** 2 + (xx - size * 0.55) ** 2) < (size * (0.2 + 0.1 * i / n)) ** 2
        img = 0.2 * top * xx / size + 0.6 * top * disc + rng.normal(0, 0.05 * top, (size, size))
        out[i] = np.clip(img, 0, top).astype(dtype)
    return out


def py2_pickle(arr):
    """The byte stream Python 2 numpy writes with pickle protocol 2 (array data as a py2 str = BINSTRING); not the
    archive format, kept as an extra format probe."""
    def s(b):
        return (b"U" + bytes([len(b)]) + b) if len(b) < 256 else (b"T" + struct.pack("<I", len(b)) + b)

    def i(n):
        return b"K" + bytes([n]) if 0 <= n < 256 else b"M" + struct.pack("<H", n)
    shape = b"".join(i(d) for d in arr.shape)
    tup = {1: b"\x85", 2: b"\x86", 3: b"\x87"}[arr.ndim]
    return (b"\x80\x02cnumpy.core.multiarray\n_reconstruct\nq\x00cnumpy\nndarray\nq\x01K\x00\x85q\x02" + s(b"b")
            + b"\x87q\x03Rq\x04(K\x01" + shape + tup + b"cnumpy\ndtype\nq\x05" + s(arr.dtype.str[1:].encode())
            + b"K\x00K\x01\x87Rq\x06(K\x03" + s(b"<") + b"NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89"
            + s(arr.tobytes()) + b"tb.")


PCK_FORMAT = "p2"  # --pck-format py2|p2|p3|p4; p2 = what the published .pck files are (pickletools on 1001866-5.pck)


def dump_pck(arr, fmt=None):
    fmt = fmt or PCK_FORMAT
    if fmt == "py2":
        return py2_pickle(arr)
    return pickle.dumps(arr, protocol=int(fmt[1:]))


def make_sources(tmp):
    mr, km = tmp / "MRNet-v1.0", tmp / "kneemri"
    arrays = {}
    for split, n, s in (("train", 0, 50), ("train", 1, 22), ("valid", 1130, 30)):
        for k, d in enumerate(("sagittal", "coronal", "axial")):
            p = mr / split / d / f"{n:04d}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            arr = phantom(s + k, 256, np.uint8, n * 10 + k)
            if n == 1 and d == "coronal":
                arr = arr[0]  # 2-D: must become an error row
            np.save(p, arr)
            arrays[f"mrnet-{n:04d}.{d}"] = arr
    meta = []
    for e, s, n in ((111, 5, 24), (111, 6, 20), (222, 3, 45)):
        p = km / "vol" / f"batch{e % 2}" / f"{e}-{s}.pck"
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = phantom(n, 320, np.uint16, e + s)
        p.write_bytes(dump_pck(arr))
        arrays[f"kneemri-{e}-{s}.sagittal"] = arr
        meta.append({"examId": e, "seriesNo": s, "aclDiagnosis": 0, "kneeLR": 1, "roiX": 0, "roiY": 0, "roiZ": 0,
                     "roiHeight": 1, "roiWidth": 1, "roiDepth": 1, "volumeFilename": f"{e}-{s}.pck"})
    pd.DataFrame(meta).to_csv(km / "metadata.csv", index=False)
    return mr, km, arrays


COMP = ["1.2.826.0.1.3680043.8.498.1", "1.2.826.0.1.3680043.8.498.2", "1.2.826.0.1.3680043.8.498.3"]
EXT_GOOD = ["mrnet-0000", "mrnet-1130", "kneemri-111-5", "kneemri-111-6", "kneemri-222-3"]
EXT_ALL = EXT_GOOD + ["mrnet-0001", "kneemri-999-1"]  # 0001 fails conversion, 999-1 is never converted


def write_labels(tmp):
    rng = np.random.default_rng(7)
    ids = COMP + EXT_ALL
    lab = pd.DataFrame(rng.uniform(-0.2, 1.2, (len(ids), len(LABELS))).round(4), columns=list(LABELS))
    lab.insert(0, ID_COL, ids)
    w = pd.DataFrame(rng.choice([0.0, 1.0], (len(ids), len(LABELS))), columns=list(LABELS))
    w.insert(0, ID_COL, ids)
    lab.to_csv(tmp / "labels.csv", index=False)
    w.to_csv(tmp / "cellw.csv", index=False)
    lab[lab[ID_COL].isin(COMP + EXT_GOOD)].to_csv(tmp / "labels_good.csv", index=False)
    return lab, w


def run_convert(tmp, ids_csv, out, *extra):
    p = subprocess.run([PY, CONVERT, "--ids", str(ids_csv), "--mrnet", str(tmp / "MRNet-v1.0"),
                        "--kneemri", str(tmp / "kneemri"), "--out", str(out), "--workers", "2", *extra],
                       capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return p


# ------------------------------------------------------------------ criteria 1-5

@check("C1 cache validity, sidecar/index/manifest params")
def t_validity(out):
    idx = pd.read_parquet(out / "index.parquet")
    assert list(idx.columns) == INDEX_COLUMNS, f"index columns differ: {set(INDEX_COLUMNS) ^ set(idx.columns)}"
    assert idx["ok"].all() and len(idx) == 3 * 2 + 3, (len(idx), idx[~idx.ok][[SERIES_COL, "error"]])
    for r in idx.itertuples(index=False):
        npy = series_path(out, r.split, getattr(r, ID_COL), getattr(r, SERIES_COL))
        assert _valid_npy(npy, CFG.px, CFG.max_slices), f"invalid npy {npy}"
        side = json.loads(npy.with_suffix(".json").read_text())
        assert side["params_sha256"] == SHA and side["ok"], npy
    assert int(idx["n_cached"].max()) <= 40 and int(idx["n_src"].max()) >= 50, "subsampling not exercised"
    man = json.loads((out / "manifest.json").read_text())
    assert man["params_sha256"] == SHA
    if REAL_V1_MANIFEST.exists():
        assert man["params_sha256"] == json.loads(REAL_V1_MANIFEST.read_text())["params_sha256"], "differs from v1"
    else:
        print("  (real v1 manifest not found; compared with configs/cache_v1.yaml only)")
    km = idx[idx[ID_COL].str.startswith("kneemri")]
    mr = idx[idx[ID_COL].str.startswith("mrnet")]
    assert np.allclose(km.spacing_mm, 0.5) and np.allclose(mr.spacing_mm, 0.625)
    assert set(km.slot) == {"sag_fs"} and set(mr.slot) == {"sag_fs", "cor_t1", "ax_fs"}
    assert (idx["rank"] == 0).all() and idx["selected"].all()


@check("C2 slot resolution via CacheIndex + load_study")
def t_slots(out):
    ci = CacheIndex(out)
    for sid in ("mrnet-0000", "mrnet-1130"):
        have = {s: ci.series_for("train", sid, s) for s in SLOTS}
        assert have["sag_fs"] == [f"{sid}.sagittal"], have
        assert have["ax_fs"] == [f"{sid}.axial"], have
        assert have["cor_t1"] == [f"{sid}.coronal"] and have["cor_fs"] == [f"{sid}.coronal"], have
        assert not have["sag_t1"] and not have["ax_t1"], have
        vols = load_study(out, "train", sid, MAIN_SLOTS, index=ci)
        assert set(vols) == {"sag_fs", "cor_fs", "ax_fs"}, set(vols)
        assert all(v.dtype == np.uint8 and v.shape[1:] == (384, 384) for v in vols.values())
    for sid in ("kneemri-111-5", "kneemri-111-6", "kneemri-222-3"):
        have = {s: ci.series_for("train", sid, s) for s in SLOTS}
        assert {s for s, v in have.items() if v} == {"sag_fs"}, have
        vols = load_study(out, "train", sid, MAIN_SLOTS, index=ci)
        assert set(vols) == {"sag_fs"}
    # split must be "train" for every external study
    assert set(ci.slots) == {"train"}


@check("C3 preprocessing equivalence with series_to_volume(spacing=160/cols)")
def t_equiv(out, arrays):
    for series, arr in arrays.items():
        if arr.ndim != 3:
            continue
        study = series.rsplit(".", 1)[0]
        npy = series_path(out, "train", study, series)
        if not npy.exists():
            continue
        n, rows, cols = arr.shape
        geom = Geometry(spacing=160.0 / cols, rows=rows, cols=cols, iop=(), ipp_centre=(0.0, 0.0, 0.0),
                        plane="Sagittal", is_3d=False, z_step=3.0, n_slices=n, obliquity=1.0)
        ref, meta = series_to_volume(list(arr), geom, CFG)
        got = np.load(npy)
        assert got.dtype == ref.dtype and got.shape == ref.shape and np.array_equal(got, ref), series
        side = json.loads(npy.with_suffix(".json").read_text())
        assert abs(side["mean_u8"] - meta.mean_u8) < 1e-9 and side["n_src"] == n and side["cols"] == cols


@check("C4 restricted unpickler")
def t_unpickle(tmp):
    arr = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    big = (np.arange(2 * 3 * 300) * 37 % 4000).astype(np.uint16).reshape(2, 3, 300)  # bytes > 0x7f in the data
    fails = {}
    for fmt in ("py2", "p2", "p3", "p4", "p5"):
        for a in (arr, big):
            p = tmp / f"ok_{fmt}.pck"
            p.write_bytes(dump_pck(a, fmt))
            try:
                got = ce.load_pck(p)
                assert np.array_equal(got, a) and got.dtype == a.dtype, fmt
            except Exception as e:  # noqa: BLE001
                fails[fmt] = f"{type(e).__name__}: {str(e)[:80]}"
    print("  formats refused/broken:", fails or "none")
    assert not {"p2", "p3", "p4"} & set(fails), fails
    flag = tmp / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {flag}",))

    class Evil2:
        def __reduce__(self):
            return (eval, ("__import__('os').getcwd()",))

    for i, obj in enumerate((Evil(), Evil2())):
        p = tmp / f"evil{i}.pck"
        p.write_bytes(pickle.dumps(obj, protocol=2))
        try:
            ce.load_pck(p)
        except pickle.UnpicklingError as e:
            assert "refusing" in str(e)
        else:
            raise AssertionError(f"evil pickle {i} loaded")
    assert not flag.exists(), "os.system ran"
    p = tmp / "list.pck"
    p.write_bytes(pickle.dumps([1, 2, 3]))
    try:
        ce.load_pck(p)
        raise AssertionError("non-array pickle accepted")
    except ValueError:
        pass
    p = tmp / "2d.pck"
    p.write_bytes(pickle.dumps(np.zeros((4, 4), np.uint16), protocol=2))
    try:
        ce.load_pck(p)
        raise AssertionError("2-D array accepted")
    except ValueError:
        pass


@check("C5 missing file / bad ID / failed series -> error row + exit 1")
def t_errors(tmp, out_bad, proc_bad):
    mr, km = tmp / "MRNet-v1.0", tmp / "kneemri"
    for sid, exc in (("mrnet-0007", FileNotFoundError), ("kneemri-999-1", FileNotFoundError),
                     ("kneemri-12-3", FileNotFoundError), ("1.2.3.4", ValueError), ("oai_1_R", ValueError)):
        try:
            ce.series_tasks([sid], mr, km)
        except exc as e:
            assert sid in str(e) or "external" in str(e), str(e)
        else:
            raise AssertionError(f"{sid}: no {exc.__name__}")
    # metadata row present but the .pck missing
    shutil.copytree(km, tmp / "km_missing")
    (tmp / "km_missing/vol/batch0/222-3.pck").unlink()
    try:
        ce.series_tasks(["kneemri-222-3"], mr, tmp / "km_missing")
        raise AssertionError("missing .pck accepted")
    except FileNotFoundError:
        pass
    # failed conversion: exit code 1, error row, other series ok
    assert proc_bad.returncode == 1, (proc_bad.returncode, proc_bad.stdout, proc_bad.stderr[-2000:])
    idx = pd.read_parquet(out_bad / "index.parquet")
    bad = idx[~idx.ok]
    assert list(bad[SERIES_COL]) == ["mrnet-0001.coronal"], bad[[SERIES_COL, "error"]]
    assert bad["error"].iloc[0].startswith("ValueError"), bad["error"].iloc[0]
    assert idx[idx[ID_COL] == "mrnet-0001"]["ok"].sum() == 2
    assert int(bad["rank"].iloc[0]) == -1 and not bad["selected"].iloc[0]
    # the run with a non-existent ID in --ids must stop before converting anything
    p = run_convert(tmp, tmp / "labels.csv", tmp / "out_missing")
    assert p.returncode != 0 and "kneemri-999-1" in (p.stderr + p.stdout), (p.returncode, p.stderr[-500:])
    assert not (tmp / "out_missing" / "index.parquet").exists()


@check("C5b every series fails -> exit 1, not a crash")
def t_all_bad(tmp):
    pd.DataFrame({ID_COL: ["mrnet-0001"]}).to_csv(tmp / "ids_one_bad.csv", index=False)
    # only the coronal of 0001 is bad; make a source where all three planes are 2-D
    mr2 = tmp / "MRNet-allbad"
    for d in ("sagittal", "coronal", "axial"):
        (mr2 / "train" / d).mkdir(parents=True)
        np.save(mr2 / "train" / d / "0001.npy", np.zeros((4, 4), np.uint8))
    p = subprocess.run([PY, CONVERT, "--ids", str(tmp / "ids_one_bad.csv"), "--mrnet", str(mr2), "--only", "mrnet",
                        "--out", str(tmp / "out_allbad"), "--workers", "1"], capture_output=True, text=True)
    assert p.returncode == 1, (p.returncode, p.stderr[-1500:])


# ------------------------------------------------------------------ criterion 6

def fake_base(tmp, ext_out):
    base_t = tmp / "base_targets"
    base_t.mkdir()
    rng = np.random.default_rng(3)
    base = pd.DataFrame({ID_COL: COMP})
    for k in LABEL_KEYS:
        base[f"{k}_target"] = rng.uniform(0, 1, len(COMP)).astype(np.float32)
        base[f"{k}_weight"] = rng.choice([0.0, 0.5, 1.0], len(COMP)).astype(np.float32)
    base.to_parquet(base_t / "full.parquet", index=False)
    (base_t / "manifest.json").write_text(json.dumps({"version": "base", "n_rows": len(COMP)}))
    (base_t / "calibration_full.json").write_text("{}")
    folds = pd.DataFrame({ID_COL: COMP + ["1.2.826.0.1.3680043.8.498.9"], "fold": ["0", "3", "1", "2"],
                          "site_key": ["Siemens|Vida|3|syngo MR XA60\\x|nl", "GE|x|1.5|y|pl", "a,b|c", "g"],
                          "lang": ["nl", "pl", "es", "en"], "is_gold": ["False", "False", "True", "True"]})
    folds.to_csv(tmp / "folds.csv", index=False, lineterminator="\n")
    # fake competition v1 cache: two train studies + one test study, built with the converter's own writer
    v1 = tmp / "v1"
    recs = []
    for j, (split, sid) in enumerate((("train", COMP[0]), ("train", COMP[1]), ("test", "1.2.826.0.1.3680043.8.498.77"))):
        for plane, fs, d in (("Sagittal", 1, "sag"), ("Coronal", 1, "cor")):
            t = (sid, f"{sid}.{j}{d}", plane, fs, "npy", tmp / f"src_{j}{d}.npy")
            np.save(t[5], phantom(20, 256, np.uint8, j))
            r = ce.convert((t, v1, CFG, 160.0, SHA))
            if split == "test":
                src = series_path(v1, "train", sid, t[1])
                dst = series_path(v1, "test", sid, t[1])
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src, dst)
                os.replace(src.with_suffix(".json"), dst.with_suffix(".json"))
                r["split"] = "test"
            recs.append(r)
    shutil.rmtree(v1 / "train" / "1.2.826.0.1.3680043.8.498.77", ignore_errors=True)
    idx = ce.index_frame(recs)
    idx.to_parquet(v1 / "index.parquet", index=False)
    (v1 / "manifest.json").write_text(json.dumps({"params_sha256": SHA, "version": "v1"}))
    return base_t, base, folds, v1, idx


@check("C6 make_inputs_ext on synthetic inputs")
def t_make_inputs(tmp, ext_out):
    base_t, base, folds, v1, v1_idx = fake_base(tmp, ext_out)
    lab, w = write_labels(tmp)
    out_t, out_f, merged = tmp / "targets_ext", tmp / "folds_ext.csv", tmp / "merged" / "v1"
    argv = ["--labels", str(tmp / "labels.csv"), "--cellw", str(tmp / "cellw.csv"), "--ext-cache", str(ext_out),
            "--base-targets", str(base_t), "--out-targets", str(out_t), "--folds", str(tmp / "folds.csv"),
            "--out-folds", str(out_f), "--v1-cache", str(v1), "--merged", str(merged)]
    mi.main(argv)
    full = pd.read_parquet(out_t / "full.parquet")
    # base rows identical (values, dtypes, order) and column order equals base
    assert list(full.columns) == list(base.columns)
    pd.testing.assert_frame_equal(full.iloc[:len(base)].reset_index(drop=True), base, check_exact=True)
    in_tbl = pd.read_parquet(base_t / "full.parquet")
    assert full.iloc[:len(base)].reset_index(drop=True).equals(in_tbl)
    ext = full.iloc[len(base):].reset_index(drop=True)
    # only fully-cached external studies (mrnet-0001 has an error row, kneemri-999-1 was never converted)
    assert sorted(ext[ID_COL]) == sorted(EXT_GOOD), sorted(ext[ID_COL])
    assert not set(ext[ID_COL]) & set(folds.loc[folds.is_gold == "True", ID_COL])
    labi, wi = lab.set_index(ID_COL), w.set_index(ID_COL)
    for col, key in zip(LABELS, LABEL_KEYS):
        t = ext[f"{key}_target"].to_numpy()
        assert ext[f"{key}_target"].dtype == np.float32 and ((t >= 0) & (t <= 1)).all(), key
        exp_t = np.clip(labi.loc[ext[ID_COL], col].to_numpy(), 0, 1).astype(np.float32)
        assert np.array_equal(t, exp_t), key
        assert np.array_equal(ext[f"{key}_weight"].to_numpy(), wi.loc[ext[ID_COL], col].to_numpy(np.float32)), key
    assert ((labi.loc[EXT_GOOD].to_numpy() < 0) | (labi.loc[EXT_GOOD].to_numpy() > 1)).any(), "clip not exercised"
    man = json.loads((out_t / "manifest.json").read_text())
    assert man["ext_rows"] == len(EXT_GOOD) and man["n_rows"] == len(base) + len(EXT_GOOD) and man["ext_rule"]
    assert (out_t / "calibration_full.json").exists()
    # folds: base rows unchanged, external folds 0-4, one exam -> one fold
    fo = pd.read_csv(out_f, dtype=str)
    assert list(fo.columns) == list(folds.columns)
    b = fo[fo[ID_COL].isin(folds[ID_COL])].sort_values(ID_COL).reset_index(drop=True)
    pd.testing.assert_frame_equal(b, folds.sort_values(ID_COL).reset_index(drop=True))
    in_lines = set((tmp / "folds.csv").read_text().splitlines()[1:])
    out_lines = set(out_f.read_text().splitlines()[1:])
    assert in_lines <= out_lines, in_lines - out_lines
    e = fo[fo[ID_COL].str.startswith(("mrnet-", "kneemri-"))].set_index(ID_COL)
    assert sorted(e.index) == sorted(EXT_GOOD) and set(e.fold) <= set("01234")
    assert (e.is_gold == "False").all() and set(e.site_key) == {"MRNET", "KNEEMRI"}
    assert e.loc["kneemri-111-5", "fold"] == e.loc["kneemri-111-6", "fold"] == str(zlib.crc32(b"kneemri-111") % 5)
    # merged cache
    mid = pd.read_parquet(merged / "index.parquet")
    ext_idx = pd.read_parquet(ext_out / "index.parquet")
    ext_idx = ext_idx[ext_idx[ID_COL].isin(EXT_GOOD)].reset_index(drop=True)  # only studies that reached the targets
    assert len(mid) == len(v1_idx) + len(ext_idx)
    pd.testing.assert_frame_equal(mid.iloc[:len(v1_idx)].reset_index(drop=True), v1_idx)
    pd.testing.assert_frame_equal(mid.iloc[len(v1_idx):].reset_index(drop=True), ext_idx[list(v1_idx.columns)],
                                  check_dtype=False)
    for r in mid[mid.ok].itertuples(index=False):
        npy = series_path(merged, r.split, getattr(r, ID_COL), getattr(r, SERIES_COL))
        assert npy.exists() and np.load(npy, mmap_mode="r").dtype == np.uint8, npy
    for s in mid.loc[mid.split == "train", ID_COL].unique():
        link = merged / "train" / s
        assert link.is_symlink() and link.resolve().is_dir(), link
    assert (merged / "test").is_symlink() and (merged / "test").resolve() == v1.resolve() / "test"
    mm = json.loads((merged / "manifest.json").read_text())
    assert mm["params_sha256"] == SHA and mm["merged_from"] == [str(v1), str(ext_out)]
    ci = CacheIndex(merged)
    assert ci.series_for("train", "kneemri-111-5", "sag_fs") and ci.series_for("train", COMP[0], "sag_fs")
    assert not ci.series_for("train", "mrnet-0001", "sag_fs"), "partially cached study must not enter the merged index"
    assert not (merged / "train" / "mrnet-0001").exists()
    assert all(mid[c].dtype == v1_idx[c].dtype for c in mid.columns), "index dtypes drift"
    # params mismatch between caches -> refused before anything is written
    bad = tmp / "ext_badsha"
    shutil.copytree(ext_out, bad, symlinks=True)
    m = json.loads((bad / "manifest.json").read_text())
    m["params_sha256"] = "0" * 64
    (bad / "manifest.json").write_text(json.dumps(m))
    try:
        mi.main(argv[:4] + ["--ext-cache", str(bad)] + argv[6:8] + ["--out-targets", str(tmp / "t_bad"),
                "--folds", str(tmp / "folds.csv"), "--out-folds", str(tmp / "f_bad.csv"), "--v1-cache", str(v1),
                "--merged", str(tmp / "merged_bad" / "v1")])
        raise RuntimeError("params mismatch accepted")
    except SystemExit as e:
        assert "cache parameters" in str(e), e
    left = [p.name for p in (tmp / "t_bad", tmp / "f_bad.csv", tmp / "merged_bad" / "v1" / "index.parquet")
            if p.exists()]
    assert not left, f"outputs written despite the params mismatch: {left}"
    # external ID colliding with a base ID -> refused
    base2 = base.copy()
    base2.loc[0, ID_COL] = "mrnet-0000"
    bt2 = tmp / "base_targets_collide"
    shutil.copytree(base_t, bt2)
    base2.to_parquet(bt2 / "full.parquet", index=False)
    try:
        mi.main(argv[:6] + ["--base-targets", str(bt2), "--out-targets", str(tmp / "t_col"),
                "--folds", str(tmp / "folds.csv"), "--out-folds", str(tmp / "f_col.csv"), "--v1-cache", str(v1),
                "--merged", str(tmp / "merged_col" / "v1")])
        raise RuntimeError("collision accepted")
    except SystemExit as e:
        assert "collides" in str(e), e
    # cache roots given as relative paths (as in the documented command line) must still give resolvable links
    rel = tmp / "merged_rel" / "v1"
    mi.main(argv[:4] + ["--ext-cache", os.path.relpath(ext_out)] + argv[6:8] + [
        "--out-targets", str(tmp / "t_rel"), "--folds", str(tmp / "folds.csv"), "--out-folds", str(tmp / "f_rel.csv"),
        "--v1-cache", os.path.relpath(v1), "--merged", str(rel)])
    dangling = [p.name for p in (rel / "train").iterdir() if not p.resolve().is_dir()]
    assert not dangling, f"relative --ext-cache/--v1-cache: {len(dangling)} dangling symlinks, e.g. {dangling[:2]}"


def main():
    global PCK_FORMAT
    keep = "--keep" in sys.argv
    if "--pck-format" in sys.argv:
        PCK_FORMAT = sys.argv[sys.argv.index("--pck-format") + 1]
    tmp = ROOT / ".tmp" / "ext_test" / f"synth_{os.getpid()}"
    tmp.mkdir(parents=True)
    t0 = time.time()
    try:
        _, _, arrays = make_sources(tmp)
        write_labels(tmp)
        out_good, out_bad = tmp / "ext_good", tmp / "ext_bad"
        p = run_convert(tmp, tmp / "labels_good.csv", out_good)
        print(p.stdout.strip())
        RESULTS.append(("C0 convert_ext exit 0 on good inputs", p.returncode == 0, p.stdout[-600:]))
        print(("PASS" if p.returncode == 0 else "FAIL") + " C0 convert_ext exit 0 on good inputs")
        if p.returncode != 0:
            print(p.stderr[-3000:])
        pd.DataFrame({ID_COL: EXT_GOOD + ["mrnet-0001"]}).to_csv(tmp / "ids_bad.csv", index=False)
        proc_bad = run_convert(tmp, tmp / "ids_bad.csv", out_bad)
        t_validity(out_good)
        t_slots(out_good)
        t_equiv(out_good, arrays)
        t_unpickle(tmp)
        t_errors(tmp, out_bad, proc_bad)
        t_all_bad(tmp)
        t_make_inputs(tmp, out_bad)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    for name, ok, msg in RESULTS:
        if not ok:
            print(f"  FAIL {name}: {msg}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
