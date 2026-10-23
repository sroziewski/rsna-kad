import pandas as pd
import pytest

from conftest import STUDY_A, STUDY_B, UID_ROOT, SyntheticData
from rsna_kad import series
from rsna_kad.constants import MAIN_SLOTS, SLOTS


def test_slot_of():
    assert series.slot_of("Sagittal", 1) == "sag_fs"
    assert series.slot_of("Coronal", "0") == "cor_t1"
    assert series.slot_of("axial", True) == "ax_fs"
    with pytest.raises(ValueError):
        series.slot_of("Oblique", 1)


def test_select_for_slot_prefers_2d_then_slices_then_series_number():
    cands = [
        {"SeriesInstanceUID": "a", "is_3d": True, "n_slices": 200, "series_number": 1},
        {"SeriesInstanceUID": "b", "is_3d": False, "n_slices": 30, "series_number": 9},
        {"SeriesInstanceUID": "c", "is_3d": False, "n_slices": 34, "series_number": 7},
        {"SeriesInstanceUID": "d", "is_3d": False, "n_slices": 34, "series_number": 5},
    ]
    assert series.select_for_slot(cands) == "d"
    assert series.select_for_slot(cands[:2]) == "b"
    assert series.select_for_slot([{"SeriesInstanceUID": "z"}]) == "z"


def _frame(synthetic: SyntheticData) -> pd.DataFrame:
    df = synthetic.series_df()
    df["n_slices"] = [synthetic.spec(u).n for u in df["SeriesInstanceUID"]]
    df["is_3d"] = [synthetic.spec(u).acq_type == "3D" for u in df["SeriesInstanceUID"]]
    df["series_number"] = [synthetic.spec(u).series_number for u in df["SeriesInstanceUID"]]
    return df


def test_assign_slots_ranks_and_falls_back(synthetic: SyntheticData):
    out = series.assign_slots(_frame(synthetic))
    assert set(out.columns) >= {"StudyInstanceUID", "slot", "SeriesInstanceUID", "rank", "selected",
                                "fallback", "native_slot"}
    a = out[out["StudyInstanceUID"] == STUDY_A]
    native_a = a[~a["fallback"]]
    assert sorted(native_a["slot"]) == ["ax_fs", "ax_t1", "cor_t1", "sag_fs"]
    assert native_a["rank"].eq(0).all()
    fb = a[a["fallback"]]
    assert list(fb["slot"]) == ["cor_fs"]
    assert list(fb["SeriesInstanceUID"]) == [UID_ROOT + "200003"]
    assert list(fb["native_slot"]) == ["cor_t1"]

    b = out[(out["StudyInstanceUID"] == STUDY_B) & (out["slot"] == "sag_fs")]
    assert list(b.sort_values("rank")["SeriesInstanceUID"]) == [UID_ROOT + "300001", UID_ROOT + "300002"]
    assert list(b.sort_values("rank")["selected"]) == [True, False]
    assert not out[(out["StudyInstanceUID"] == STUDY_B) & out["fallback"]].shape[0]


def test_assign_slots_without_rank_columns(synthetic: SyntheticData):
    out = series.assign_slots(synthetic.series_df(), slots=MAIN_SLOTS)
    assert set(out["slot"]) <= set(MAIN_SLOTS)
    # ties broken by SeriesInstanceUID, deterministically
    b = out[(out["StudyInstanceUID"] == STUDY_B) & (out["slot"] == "sag_fs")].sort_values("rank")
    assert list(b["SeriesInstanceUID"]) == [UID_ROOT + "300001", UID_ROOT + "300002"]


def test_assign_slots_skips_failed_series(synthetic: SyntheticData):
    df = _frame(synthetic)
    df["ok"] = df["SeriesInstanceUID"] != UID_ROOT + "300001"
    out = series.assign_slots(df)
    b = out[(out["StudyInstanceUID"] == STUDY_B) & (out["slot"] == "sag_fs")]
    assert list(b["SeriesInstanceUID"]) == [UID_ROOT + "300002"] and list(b["rank"]) == [0]


def test_slot_table(synthetic: SyntheticData):
    table = series.slot_table(series.assign_slots(_frame(synthetic), SLOTS))
    assert table[STUDY_A]["cor_fs"] == [UID_ROOT + "200003"]
    assert table[STUDY_B]["sag_fs"] == [UID_ROOT + "300001", UID_ROOT + "300002"]
    assert "ax_t1" not in table[STUDY_B]
