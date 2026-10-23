"""Tests for qa/bprefit/make_members.py on synthetic qoof_driver-style outputs (tmp_path):

  PYTHONDONTWRITEBYTECODE=1 python -m pytest qa/bprefit -q -p no:cacheprovider
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("make_members", HERE / "make_members.py")
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

ID = "StudyInstanceUID"
LABELS = [f"finding_{k:02d}" for k in range(12)]
WP = {f: 0.1 + 0.05 * k for k, f in enumerate(LABELS)}


def frame(ids, seed):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.rand(len(ids), len(LABELS)), index=pd.Index(ids, name=ID), columns=LABELS)


# ---------------------------------------------------------------- merge_olp

def test_merge_olp_formula_both_present():
    ids = ["1.2.3", "1.2.4"]
    s, c = frame(ids, 1), frame(ids, 2)
    olp, n = mm.merge_olp(s, c, WP, LABELS)
    assert n == 0
    for f in LABELS:
        np.testing.assert_allclose(olp[f].values, WP[f] * s[f].values + (1 - WP[f]) * c[f].values)


def test_merge_olp_c_missing_uses_s_not_counted():
    ids = ["a", "b", "c"]
    s, c = frame(ids, 1), frame(ids, 2)
    c.loc["b"] = np.nan
    c.loc["c", LABELS[3]] = np.nan
    olp, n = mm.merge_olp(s, c, WP, LABELS)
    assert n == 0
    np.testing.assert_allclose(olp.loc["b"].values, s.loc["b"].values)
    np.testing.assert_allclose(olp.loc["c", LABELS[3]], s.loc["c", LABELS[3]])
    f = LABELS[4]
    np.testing.assert_allclose(olp.loc["c", f], WP[f] * s.loc["c", f] + (1 - WP[f]) * c.loc["c", f])


def test_merge_olp_s_missing_uses_c_and_counts():
    ids = ["a", "b", "c", "d"]
    s, c = frame(ids, 1), frame(ids, 2)
    s.loc[["b", "d"]] = np.nan
    olp, n = mm.merge_olp(s, c, WP, LABELS)
    assert n == 2
    np.testing.assert_allclose(olp.loc[["b", "d"]].values, c.loc[["b", "d"]].values)
    f = LABELS[0]
    np.testing.assert_allclose(olp.loc["a", f], WP[f] * s.loc["a", f] + (1 - WP[f]) * c.loc["a", f])
    assert not olp.isna().any().any()


def test_merge_olp_s_reindexed_row_absent():
    ids = ["a", "b"]
    s = frame(["a"], 1).reindex(ids)
    c = frame(ids, 2)
    olp, n = mm.merge_olp(s, c, WP, LABELS)
    assert n == 1
    np.testing.assert_allclose(olp.loc["b"].values, c.loc["b"].values)


def test_merge_olp_both_missing_stays_nan():
    ids = ["a", "b"]
    s, c = frame(ids, 1), frame(ids, 2)
    s.loc["b"] = np.nan
    c.loc["b"] = np.nan
    olp, n = mm.merge_olp(s, c, WP, LABELS)
    assert olp.loc["b"].isna().all()


# ---------------------------------------------------------------- members

def write_work(work, n_oof=(3, 2, 2, 2, 2), n_gold=3, s_missing=(), mth_ids=None):
    work.mkdir(parents=True, exist_ok=True)
    tags = [(f"oof_f{k}", n) for k, n in enumerate(n_oof)] + [("gold", n_gold)]
    seed = 0
    allids = {}
    for tag, n in tags:
        ids = [f"1.2.840.{tag}.{i:05d}" for i in range(n)]
        allids[tag] = ids
        for m in ("MTR", "MTH", "OLPS", "OLPC"):
            seed += 1
            these = ids if not (m == "MTH" and mth_ids) else mth_ids(ids)
            df = frame(these, seed)
            if m == "OLPS":
                df = df.drop([i for i in s_missing if i in df.index])
            df.insert(0, "extra_col", 7)
            # shuffle the columns to check that members() selects by label
            df = df[["extra_col"] + LABELS[::-1]]
            df.to_csv(work / f"{tag}_{m}_foldmean.csv", index_label=ID)
    return allids


def test_members_frames(tmp_path):
    ids = write_work(tmp_path / "w", s_missing=["1.2.840.oof_f0.00001"])
    res, report = mm.members(tmp_path / "w", WP, LABELS)
    assert set(res) == {"MTR", "MTH", "OLP"}
    oof = [i for k in range(5) for i in ids[f"oof_f{k}"]]
    for m, df in res.items():
        assert list(df.columns) == ["set"] + LABELS
        assert df.index.is_unique
        assert not df.isna().any().any()
        assert list(df.index) == oof + ids["gold"]
        assert (df.loc[oof, "set"] == "oof").all()
        assert (df.loc[ids["gold"], "set"] == "gold58").all()
    assert report["oof_f0"] == dict(rows=3, olp_coronal_only=1)
    assert report["gold"] == dict(rows=3, olp_coronal_only=0)
    mtr = mm.read(tmp_path / "w" / "oof_f1_MTR_foldmean.csv")
    pd.testing.assert_frame_equal(res["MTR"].loc[mtr.index, LABELS], mtr[LABELS])
    s = mm.read(tmp_path / "w" / "gold_OLPS_foldmean.csv")[LABELS]
    c = mm.read(tmp_path / "w" / "gold_OLPC_foldmean.csv")[LABELS]
    f = LABELS[5]
    np.testing.assert_allclose(res["OLP"].loc[s.index, f].values, WP[f] * s[f].values + (1 - WP[f]) * c[f].values)
    c0 = mm.read(tmp_path / "w" / "oof_f0_OLPC_foldmean.csv")[LABELS]
    np.testing.assert_allclose(res["OLP"].loc["1.2.840.oof_f0.00001", LABELS].values.astype(float),
                               c0.loc["1.2.840.oof_f0.00001"].values)


def test_members_ids_kept_as_strings(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    for tag in [f"oof_f{k}" for k in range(5)] + ["gold"]:
        ids = [f"00{k}{tag[-1]}" for k in range(2)]  # leading zeros would be lost if read as int
        for m in ("MTR", "MTH", "OLPS", "OLPC"):
            frame(ids, 3).to_csv(work / f"{tag}_{m}_foldmean.csv", index_label=ID)
    res, _ = mm.members(work, WP, LABELS)
    assert all(i.startswith("00") for i in res["MTR"].index)


def test_members_asserts_on_mth_index_mismatch(tmp_path):
    write_work(tmp_path / "w", mth_ids=lambda ids: ids[::-1])
    with pytest.raises(AssertionError):
        mm.members(tmp_path / "w", WP, LABELS)


def test_members_asserts_on_duplicate_ids_across_folds(tmp_path):
    work = tmp_path / "w"
    write_work(work)
    for m in ("MTR", "MTH", "OLPS", "OLPC"):
        df = pd.read_csv(work / f"oof_f0_{m}_foldmean.csv", dtype={ID: str})
        df.loc[0, ID] = "1.2.840.oof_f1.00000"
        df.to_csv(work / f"oof_f0_{m}_foldmean.csv", index=False)
    with pytest.raises(AssertionError):
        mm.members(work, WP, LABELS)


def test_members_asserts_on_nan(tmp_path):
    work = tmp_path / "w"
    write_work(work)
    df = pd.read_csv(work / "oof_f2_MTR_foldmean.csv", dtype={ID: str})
    df.loc[1, LABELS[2]] = np.nan
    df.to_csv(work / "oof_f2_MTR_foldmean.csv", index=False)
    with pytest.raises(AssertionError):
        mm.members(work, WP, LABELS)


# ---------------------------------------------------------------- main

def inputs(tmp_path, n_oof, n_gold, s_missing=()):
    work = tmp_path / "work"
    ids = write_work(work, n_oof=n_oof, n_gold=n_gold, s_missing=s_missing)
    (tmp_path / "w.json").write_text(json.dumps({"sagittal_weight": WP, "other": 1}))
    pd.DataFrame(columns=[ID] + LABELS).to_csv(tmp_path / "labels.csv", index=False)
    out = tmp_path / "out"
    argv = ["make_members.py", "--work", str(work), "--weights", str(tmp_path / "w.json"),
            "--labels", str(tmp_path / "labels.csv"), "--out", str(out)]
    return argv, out, ids


def run_main(argv):
    old, sys.argv = sys.argv, argv
    try:
        mm.main()
    finally:
        sys.argv = old


def test_main_exits_on_wrong_counts(tmp_path):
    argv, out, _ = inputs(tmp_path, n_oof=(3, 2, 2, 2, 2), n_gold=3)
    with pytest.raises(SystemExit) as e:
        run_main(argv)
    assert e.value.code not in (0, None)
    assert "unexpected rows" in str(e.value.code)
    assert not out.exists()


def test_main_exits_when_gold_count_off_by_one(tmp_path):
    argv, out, _ = inputs(tmp_path, n_oof=(870, 870, 870, 870, 869), n_gold=57)
    with pytest.raises(SystemExit) as e:
        run_main(argv)
    assert e.value.code not in (0, None)
    assert not out.exists()


def test_main_rejects_label_csv_without_12_findings(tmp_path):
    argv, _, _ = inputs(tmp_path, n_oof=(3, 2, 2, 2, 2), n_gold=3)
    pd.DataFrame(columns=[ID] + LABELS[:11]).to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(AssertionError):
        run_main(argv)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("full")
    argv, out, ids = inputs(tmp, n_oof=(870, 870, 870, 870, 869), n_gold=58, s_missing=["1.2.840.oof_f3.00007"])
    run_main(argv)
    return out, ids


def test_main_writes_member_files(full_run):
    out, _ = full_run
    assert sorted(p.name for p in out.iterdir()) == sorted(mm.NAMES.values())
    for name in mm.NAMES.values():
        header = (out / name).read_text().split("\n")[0]
        assert header == ",".join([ID, "set"] + LABELS)


def test_outputs_readable_by_fit_stacker5_logic(full_run):
    out, ids = full_run
    oof_ids = [i for k in range(5) for i in ids[f"oof_f{k}"]]
    for name in ("member_mt_r50.csv", "member_mt_hires.csv", "member_ol_plane.csv"):
        df = pd.read_csv(out / name, dtype={ID: str}).set_index(ID)
        sel = df[df.set == "oof"]
        assert len(sel) == 4349 and (df.set == "gold58").sum() == 58
        assert list(sel.index) == oof_ids
        assert all(f in df.columns for f in LABELS)
        assert sel[LABELS].dtypes.map(lambda t: np.issubdtype(t, np.floating)).all()
        assert not df[LABELS].isna().any().any()
    olp = pd.read_csv(out / "member_ol_plane.csv", dtype={ID: str}).set_index(ID)
    c = mm.read(out.parent / "work" / "oof_f3_OLPC_foldmean.csv")[LABELS]
    np.testing.assert_allclose(olp.loc["1.2.840.oof_f3.00007", LABELS].values.astype(float),
                               c.loc["1.2.840.oof_f3.00007"].values, rtol=1e-12)
