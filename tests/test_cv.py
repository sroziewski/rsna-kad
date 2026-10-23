from pathlib import Path

import pandas as pd
import pytest

from rsna_kad import cv
from rsna_kad.constants import ID_COL

REPO = Path(__file__).resolve().parents[1]
FOLDS_V1 = REPO / "data" / "folds" / "folds_v1.csv"
# Pinned after `scripts/make_folds.py` produced the file on the server (seed 2026).
FOLDS_V1_MD5 = "4dc54b74429f07051af2175e7bb35442"


def test_make_folds_groups_and_stratifies(study_summary_csv: Path):
    folds = cv.make_folds(study_summary_csv, n_splits=5, seed=2026)
    assert list(folds.columns) == cv.FOLD_COLUMNS
    assert len(folds) == 400 and folds[ID_COL].is_unique
    assert sorted(folds["fold"].unique()) == [0, 1, 2, 3, 4]
    assert (folds.groupby("site_key")["fold"].nunique() == 1).all()
    assert folds["is_gold"].sum() == 30
    sizes = folds["fold"].value_counts(normalize=True)
    assert sizes.between(0.10, 0.30).all()
    again = cv.make_folds(study_summary_csv, n_splits=5, seed=2026)
    pd.testing.assert_frame_equal(folds, again)
    assert not folds["fold"].equals(cv.make_folds(study_summary_csv, n_splits=5, seed=1)["fold"])
    assert sizes.between(0.17, 0.23).all()
    assert folds.groupby("fold")["is_gold"].sum().between(3, 9).all()


def test_make_folds_sklearn_method(study_summary_csv: Path):
    folds = cv.make_folds(study_summary_csv, n_splits=5, seed=2026, method="sklearn")
    assert (folds.groupby("site_key")["fold"].nunique() == 1).all()
    assert sorted(folds["fold"].unique()) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        cv.make_folds(study_summary_csv, method="nope")


def test_check_folds_reports_violations(study_summary_csv: Path):
    folds = cv.make_folds(study_summary_csv, n_splits=5, seed=2026)
    broken = folds.copy()
    broken.loc[broken.index[:3], "fold"] = (broken.loc[broken.index[:3], "fold"] + 1) % 5
    problems = cv.check_folds(broken, gold_range=(0, 400), frac_range=(0.0, 1.0))
    assert any("span several folds" in p for p in problems)
    assert cv.check_folds(folds, gold_range=(0, 400), frac_range=(0.0, 1.0)) == []


def test_write_and_load_roundtrip(study_summary_csv: Path, tmp_path: Path):
    folds = cv.make_folds(study_summary_csv, n_splits=5, seed=2026)
    md5 = cv.write_folds(folds, tmp_path / "f.csv")
    assert md5 == cv.write_folds(folds, tmp_path / "g.csv")
    back = cv.load_folds(tmp_path / "f.csv")
    pd.testing.assert_frame_equal(back, folds.sort_values(ID_COL).reset_index(drop=True))
    assert (tmp_path / "f.csv").read_bytes().count(b"\r") == 0


def test_folds_v1_pinned():
    assert FOLDS_V1.exists(), "run scripts/make_folds.py on the server and pull data/folds"
    assert cv.md5_of(FOLDS_V1) == FOLDS_V1_MD5
    folds = cv.load_folds(FOLDS_V1)
    assert len(folds) == 4407 and folds[ID_COL].is_unique
    assert folds["is_gold"].sum() == 58
    assert folds["site_key"].nunique() == 123
    assert cv.check_folds(folds) == []
