import numpy as np
import pandas as pd
import pytest

from conftest import STUDY_A, STUDY_B, SyntheticData
from rsna_kad import dicom, laterality
from rsna_kad.constants import ID_COL


def test_parse_tag_and_description():
    assert laterality.parse_laterality_tag("L") == "L"
    assert laterality.parse_laterality_tag(" right ") == "R"
    assert laterality.parse_laterality_tag("") == "" and laterality.parse_laterality_tag(None) == ""
    assert laterality.parse_desc_side("RT_COR_PD_FS") == "R"
    assert laterality.parse_desc_side("LT_SAG_T1") == "L"
    assert laterality.parse_desc_side("COR T1 L") == "L"
    assert laterality.parse_desc_side("Sag PD FSE FS R") == "R"
    assert laterality.parse_desc_side("pd_tse_fs_sag_320") == ""
    assert laterality.parse_desc_side("DummySeriesDesc!") == ""
    assert laterality.parse_desc_side("L R mixed") == ""
    assert laterality.parse_desc_side("SAG PD FS") == ""   # 'S' inside a word never matches


def test_manufacturer_family():
    assert laterality.manufacturer_family("Siemens Healthineers") == "siemens"
    assert laterality.manufacturer_family("SIEMENS") == "siemens"
    assert laterality.manufacturer_family("Philips Healthcare") == "philips"
    assert laterality.manufacturer_family("Philips Medical Systems") == "philips"
    assert laterality.manufacturer_family("GE MEDICAL SYSTEMS") == "ge"
    assert laterality.manufacturer_family("GEHC") == "ge"
    assert laterality.manufacturer_family("TOSHIBA") == laterality.manufacturer_family("CANON_MEC") == "canon"
    assert laterality.manufacturer_family("FUJIFILM Healthcare Corporation") == "hitachi"
    assert laterality.manufacturer_family("") == "unknown"


def test_majority_and_aggregate():
    rows = [{"side_tag": "L", "side_desc": "", "plane": "Sagittal", "centre_x": 50.0},
            {"side_tag": "R", "side_desc": "", "plane": "Axial", "centre_x": 60.0},
            {"side_tag": "L", "side_desc": "L", "plane": "Coronal", "centre_x": 70.0},
            {"side_tag": "L", "side_desc": "", "plane": "Axial", "centre_x": 62.0}]
    tag, conf, desc, cx = laterality.aggregate_series(rows)
    assert (tag, conf, desc) == ("L", laterality.TAG_CONF_CONTRADICTORY, "L")
    assert cx == pytest.approx(61.0)  # axial preferred over coronal/sagittal
    tag, conf, _, cx = laterality.aggregate_series(rows[:1])
    assert (tag, conf) == ("L", laterality.TAG_CONF_CONSISTENT) and cx == 50.0
    tag, conf, desc, cx = laterality.aggregate_series([{"side_tag": "", "side_desc": "", "plane": "Axial",
                                                        "centre_x": float("nan")}])
    assert (tag, conf, desc) == ("", 0.0, "") and np.isnan(cx)


def _study_frame(n_siemens=60, n_ge=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_siemens):
        side = "L" if i % 2 else "R"
        cx = (40 + rng.random() * 30) * (1 if side == "L" else -1)
        if i % 50 == 49:
            cx = -cx  # one wrong tag per 50
        rows.append({ID_COL: f"s{i}", "manufacturer": "SIEMENS", "side_tag": side, "tag_conf": 0.95,
                     "side_desc": "", "centre_x": cx})
    for i in range(n_ge):
        side = "L" if i % 2 else "R"
        rows.append({ID_COL: f"g{i}", "manufacturer": "GE", "side_tag": side, "tag_conf": 0.95,
                     "side_desc": "", "centre_x": rng.random() * 40 - 20})  # no information
    rows.append({ID_COL: "u1", "manufacturer": "SIEMENS", "side_tag": "", "tag_conf": 0.0,
                 "side_desc": "", "centre_x": 55.0})
    rows.append({ID_COL: "u2", "manufacturer": "GE", "side_tag": "", "tag_conf": 0.0,
                 "side_desc": "R", "centre_x": 55.0})
    rows.append({ID_COL: "u3", "manufacturer": "GE", "side_tag": "", "tag_conf": 0.0,
                 "side_desc": "", "centre_x": 55.0})
    rows.append({ID_COL: "c1", "manufacturer": "SIEMENS", "side_tag": "R", "tag_conf": 0.6,
                 "side_desc": "", "centre_x": 55.0})
    return pd.DataFrame(rows)


def test_fit_geom_convention_and_estimate():
    df = _study_frame()
    rules = laterality.fit_geom_convention(df, min_studies=30, min_agreement=0.95)
    assert rules["siemens"].sign == 1 and rules["siemens"].trusted and rules["siemens"].n == 60
    assert rules["siemens"].agreement == pytest.approx(59 / 60)
    assert not rules["ge"].trusted and rules["ge"].agreement < 0.8
    out = laterality.estimate_sides(df, rules).set_index(ID_COL)
    assert out.loc["u1", ["side_geom", "side_est"]].tolist() == ["L", "L"]
    assert out.loc["u1", "side_conf"] == pytest.approx(59 / 60)
    assert out.loc["u2", "side_est"] == "R" and out.loc["u2", "side_conf"] == laterality.DESC_CONF
    assert out.loc["u3", "side_est"] == "unknown" and out.loc["u3", "side_conf"] == 0.0
    assert out.loc["c1", "side_est"] == "R" and out.loc["c1", "side_conf"] == pytest.approx(0.6)
    assert out.loc["s1", "side_est"] == "L" and out.loc["s1", "side_conf"] == pytest.approx(0.95)
    # contradictory tags are kept but geometry never overrides an explicit tag
    assert out.loc["s49", "side_geom"] != out.loc["s49", "side_tag"]
    assert out.loc["s49", "side_est"] == out.loc["s49", "side_tag"]


def test_fit_with_flipped_convention():
    df = _study_frame()
    df.loc[df["manufacturer"] == "SIEMENS", "centre_x"] *= -1
    rules = laterality.fit_geom_convention(df)
    assert rules["siemens"].sign == -1 and rules["siemens"].trusted
    out = laterality.estimate_sides(df, rules).set_index(ID_COL)
    assert out.loc["u1", "centre_x"] == -55.0 and out.loc["u1", "side_geom"] == "L"
    assert (out.loc[out.index.str.startswith("s"), "side_geom"]
            == out.loc[out.index.str.startswith("s"), "side_tag"]).mean() == pytest.approx(59 / 60)


def test_estimate_side_on_synthetic_study(synthetic: SyntheticData):
    for study, expect_tag, expect_conf in ((STUDY_A, "L", laterality.TAG_CONF_CONTRADICTORY),
                                           (STUDY_B, "", 0.0)):
        headers = {s.uid: [dicom.read_header(p) for p in s.paths] for s in synthetic.specs[study]}
        series_df = synthetic.series_df()
        side = laterality.estimate_side(headers, series_df[series_df[ID_COL] == study])
        assert side.side_tag == expect_tag and side.tag_conf == expect_conf
        assert side.side_geom == ""  # no convention passed
    side_a = laterality.estimate_side(
        {s.uid: [dicom.read_header(p) for p in s.paths] for s in synthetic.specs[STUDY_A]},
        synthetic.series_df())
    assert side_a.side_desc == "L" and side_a.side_est == "L" and side_a.centre_x == pytest.approx(60.0)
    rules = {"siemens": laterality.GeomRule(1, 0.99, 100, True)}
    side_b = laterality.estimate_side(
        {s.uid: [dicom.read_header(p) for p in s.paths] for s in synthetic.specs[STUDY_B]},
        synthetic.series_df(), rules)
    assert side_b.centre_x == pytest.approx(-55.0)
    assert side_b.side_geom == "R" and side_b.side_est == "R" and side_b.side_conf == pytest.approx(0.99)


def test_agreement_report():
    df = _study_frame()
    rules = laterality.fit_geom_convention(df)
    rep = laterality.agreement_report(df, rules).set_index("manufacturer")
    assert rep.loc["siemens", "n_tagged"] == 60 and rep.loc["siemens", "trusted"]
    assert rep.loc["ge", "n_studies"] == 42
