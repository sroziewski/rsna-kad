import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import STUDY_A, STUDY_B, UID_ROOT, SyntheticData
from rsna_kad import cache
from rsna_kad.config import (CacheConfig, TrainConfig, from_dict, load_cache_config, load_config,
                             params_sha256)
from rsna_kad.constants import ID_COL, MAIN_SLOTS, SERIES_COL, SLOTS

REPO = Path(__file__).resolve().parents[1]


def test_cache_config_yaml_matches_defaults():
    cfg = load_cache_config(REPO / "configs" / "cache_v1.yaml")
    assert cfg == CacheConfig()
    assert (cfg.crop_mm, cfg.px, cfg.max_slices, cfg.z_target_mm) == (130.0, 384, 40, 3.0)
    assert params_sha256(cfg) == params_sha256(CacheConfig())


def test_config_unknown_key_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run_id: x\ncache:\n  crop_mm: 120\n  typo_key: 1\n")
    with pytest.raises(KeyError, match="typo_key"):
        load_config(bad)
    good = tmp_path / "good.yaml"
    good.write_text("run_id: x\ncache:\n  crop_mm: 120\nmodel:\n  slots: [sag_fs, ax_fs]\n")
    cfg = load_config(good)
    assert isinstance(cfg, TrainConfig) and cfg.cache.crop_mm == 120 and cfg.cache.px == 384
    assert cfg.model.slots == ("sag_fs", "ax_fs")
    with pytest.raises(KeyError):
        from_dict(CacheConfig, {"px": 1, "nope": 2})


@pytest.fixture(scope="module")
def built(synthetic: SyntheticData, tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("cache") / "v1"
    cfg = CacheConfig()
    train = cache.build_cache(synthetic.data_dir, out, "train", cfg, workers=2)
    test = cache.build_cache(synthetic.data_dir, out, "test", cfg, workers=2)
    return out, cfg, train, test


def test_index_contract(built):
    out, cfg, train, test = built
    assert len(train) == 9 and train["ok"].all()
    assert len(test) == 5 and test["ok"].all()
    for col in ["split", ID_COL, SERIES_COL, "plane", "fs", "slot", "rank", "selected", "is_3d", "n_src",
                "n_cached", "spacing_mm", "fov_mm", "z_step_mm", "crop_mm", "px", "p_lo", "p_hi",
                "truncated_slices", "side_tag", "side_desc", "side_geom", "side_est", "side_conf", "ok", "error"]:
        assert col in train.columns, col
    idx = cache.read_index(out)
    assert len(idx) == 14 and sorted(idx["split"].unique()) == ["test", "train"]
    row = idx[idx[SERIES_COL] == UID_ROOT + "200001"].iloc[0]
    assert row["truncated_slices"] == 1 and row["ok"] and row["n_cached"] == 30
    row3d = idx[idx[SERIES_COL] == UID_ROOT + "200002"].iloc[0]
    assert row3d["is_3d"] and row3d["n_src"] == 120 and row3d["n_cached"] == 40
    assert idx["side_conf"].dtype == np.float32
    manifest = cache.read_manifest(out)
    assert manifest["params_sha256"] == params_sha256(cfg)
    assert manifest["n_series"] == 14 and manifest["n_ok"] == 14
    assert manifest["splits"]["train"]["n_series"] == 9


def test_npy_files_and_slots(built):
    out, cfg, train, _ = built
    for r in train.itertuples(index=False):
        arr = cache.load_series(out, "train", getattr(r, ID_COL), getattr(r, SERIES_COL))
        assert arr.dtype == np.uint8 and arr.shape[1:] == (384, 384) and 1 <= arr.shape[0] <= 40
        assert arr.shape[0] == r.n_cached
        assert 20 <= arr.mean() <= 200
    a = train[train[ID_COL] == STUDY_A].set_index(SERIES_COL)
    assert a.loc[UID_ROOT + "200003", "slot"] == "cor_t1" and a.loc[UID_ROOT + "200003", "selected"]
    b = train[train[ID_COL] == STUDY_B].set_index(SERIES_COL)
    assert b.loc[UID_ROOT + "300001", "rank"] == 0 and b.loc[UID_ROOT + "300002", "rank"] == 1
    assert not b.loc[UID_ROOT + "300002", "selected"]


def test_laterality_columns(built):
    out, cfg, train, _ = built
    a = train[train[ID_COL] == STUDY_A].iloc[0]
    assert a["side_tag"] == "L" and a["side_desc"] == "L" and a["side_est"] == "L"
    assert a["side_conf"] == pytest.approx(0.6)  # L, L, R tags → contradictory
    b = train[train[ID_COL] == STUDY_B].iloc[0]
    assert b["side_tag"] == "" and b["side_est"] == "unknown" and b["side_conf"] == 0.0
    assert b["side_geom"] == ""  # too few tagged studies to trust any geometric convention


def test_load_study_with_fallback_and_sibling(built):
    out, cfg, _, _ = built
    vols = cache.load_study(out, "train", STUDY_A, MAIN_SLOTS)
    assert set(vols) == {"sag_fs", "cor_fs", "ax_fs"}
    assert vols["cor_fs"].shape[0] == 20  # the coronal T1 lent to the missing FS slot
    vols_b = cache.load_study(out, "train", STUDY_B, SLOTS)
    assert set(vols_b) == {"sag_fs", "ax_fs", "cor_fs", "sag_t1"}
    assert vols_b["sag_fs"].shape[0] == 12
    picks = {cache.load_study(out, "train", STUDY_B, ["sag_fs"], sibling_rng=np.random.default_rng(i),
                              sibling_p=1.0)["sag_fs"].shape[0] for i in range(5)}
    assert picks == {8}
    assert cache.load_study(out, "test", STUDY_B, ["sag_fs"])["sag_fs"].shape[0] == 12


def test_resume_skips_valid_series(built, synthetic: SyntheticData, caplog):
    out, cfg, train, _ = built
    npy = cache.series_path(out, "train", STUDY_B, UID_ROOT + "300003")
    before = npy.stat().st_mtime_ns
    sidecar = json.loads(npy.with_suffix(".json").read_text())
    assert sidecar["ok"] and sidecar["params_sha256"] == params_sha256(cfg)
    again = cache.build_cache(synthetic.data_dir, out, "train", cfg, workers=2)
    assert npy.stat().st_mtime_ns == before
    pd.testing.assert_frame_equal(again.reset_index(drop=True), train.reset_index(drop=True))
    # a damaged file is rebuilt
    npy.write_bytes(b"garbage")
    cache.build_cache(synthetic.data_dir, out, "train", cfg, workers=2)
    assert np.load(npy).shape == (6, 384, 384)
    # a changed parameter set forces a rebuild of everything
    other = CacheConfig(px=128)
    rebuilt = cache.build_cache(synthetic.data_dir, out, "train", other, workers=2)
    assert np.load(npy).shape == (6, 128, 128) and rebuilt["ok"].all()
    cache.build_cache(synthetic.data_dir, out, "train", cfg, workers=2)
    assert np.load(npy).shape == (6, 384, 384)


def test_missing_series_dir_is_an_error_row(built, synthetic: SyntheticData, tmp_path: Path):
    out, cfg, _, _ = built
    data = tmp_path / "data"
    (data / "train_series").mkdir(parents=True)
    pd.DataFrame([[STUDY_A, "missing", 1, 1, "Axial"]],
                 columns=["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression",
                          "Anatomical_Plane"]).to_csv(data / "train_series.csv", index=False)
    df = cache.build_cache(data, tmp_path / "c", "train", cfg, workers=1)
    assert len(df) == 1 and not df["ok"].iloc[0] and "FileNotFoundError" in df["error"].iloc[0]
    assert df["rank"].iloc[0] == -1


def test_verify_cache(built):
    out, cfg, train, test = built
    report = cache.verify_cache(out, workers=2)
    assert report["passed"]
    assert report["n_index_rows"] == 14 and report["n_ok"] == 14 and report["n_files_bad"] == 0
    assert report["n_truncated_series"] == 1 and report["truncated_slices_total"] == 1
    assert report["mean_in_range_frac"] == 1.0
    assert 1 <= report["n_slices_min"] and report["n_slices_max"] <= 40
    assert report["slot_coverage"]["ax_fs"] == 1.0 and report["slot_coverage"]["cor_fs"] == 1.0
    assert report["total_bytes"] == sum(
        os.path.getsize(cache.series_path(out, r.split, getattr(r, ID_COL), getattr(r, SERIES_COL)))
        for r in cache.read_index(out).itertuples(index=False))
    npy = cache.series_path(out, "test", STUDY_B, UID_ROOT + "300005")
    np.save(npy, np.zeros((2, 10, 10), np.uint8))
    bad = cache.verify_cache(out, workers=1)
    assert not bad["passed"] and bad["n_files_bad"] == 1
