"""StudyDataset over a tiny hand-written cache (npy volumes + index.parquet), no DICOM."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from rsna_kad import cache
from rsna_kad.config import ModelConfig, TrainConfig
from rsna_kad.constants import ID_COL, LABEL_KEYS, SERIES_COL
from rsna_kad.data import (StudyDataset, augment_params, augment_volume, collate_studies, load_targets,
                           read_triplets, sample_centres, slot_view_mm, synthetic_targets, triplet_indices,
                           view_params, warp_slices)
from rsna_kad.series import slot_of

PX = 48
STUDIES = ["S1", "S2", "S3", "S4"]
# (study, series, plane, fs, n_slices, series_number); S2 lacks any coronal series, S3 has
# only T1 sagittal/coronal (fallback slots), S1 has a sagittal FS sibling.
SERIES = [
    ("S1", "S1a", "Sagittal", 1, 12, 3), ("S1", "S1b", "Sagittal", 1, 8, 7), ("S1", "S1c", "Coronal", 1, 10, 4),
    ("S1", "S1d", "Axial", 1, 6, 5),
    ("S2", "S2a", "Sagittal", 1, 9, 2), ("S2", "S2b", "Axial", 1, 7, 3),
    ("S3", "S3a", "Sagittal", 0, 5, 2), ("S3", "S3b", "Coronal", 0, 5, 3), ("S3", "S3c", "Axial", 1, 4, 4),
    ("S4", "S4a", "Sagittal", 1, 11, 1), ("S4", "S4b", "Coronal", 1, 9, 2), ("S4", "S4c", "Axial", 1, 8, 3),
    ("S5", "S5a", "Sagittal", 1, 7, 1), ("S5", "S5b", "Coronal", 1, 6, 2), ("S5", "S5c", "Axial", 1, 5, 3),
]


def make_tiny_cache(root: Path, px: int = PX) -> Path:
    """<root>/v1/train/<study>/<series>.npy + index.parquet with the columns CacheIndex needs."""
    out = root / "v1"
    rows = []
    for study, series, plane, fs, n, num in SERIES:
        rng = np.random.default_rng(abs(hash(series)) % 2**32)
        yy, xx = np.mgrid[0:px, 0:px]
        base = (80 + 60 * np.sin(xx / 6.0) + 40 * np.cos(yy / 9.0))
        vol = np.stack([np.clip(base + 4 * i + rng.normal(0, 8, (px, px)), 0, 255) for i in range(n)]).astype(np.uint8)
        path = cache.series_path(out, "train", study, series)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vol)
        rows.append({"split": "train", ID_COL: study, SERIES_COL: series, "plane": plane, "fs": fs,
                     "slot": slot_of(plane, fs), "rank": 0, "selected": True, "is_3d": False,
                     "n_src": n, "n_cached": n, "spacing_mm": 0.34, "fov_mm": 130.0, "z_step_mm": 3.0,
                     "crop_mm": 130.0, "px": px, "p_lo": 1.0, "p_hi": 200.0, "truncated_slices": 0,
                     "side_tag": "", "side_desc": "", "side_geom": "", "side_est": "unknown",
                     "side_conf": np.float32(0), "ok": True, "error": "", "manufacturer": "X",
                     "series_number": num, "rows": px, "cols": px, "centre_x": 0.0, "mean_u8": 100.0,
                     "z_step_cached_mm": 3.0, "tag_conf": 0.0, "plane_iop": plane})
    pd.DataFrame(rows)[cache.INDEX_COLUMNS].to_parquet(out / cache.INDEX_NAME, index=False)
    (out / cache.MANIFEST_NAME).write_text('{"version": "v1"}')
    return out


def tiny_cfg(**kw) -> TrainConfig:
    model = ModelConfig(backbone="resnet18", img_px=32, slots=("sag_fs", "cor_fs", "ax_fs"), k_train=4,
                        k_infer=6, attn_hidden=16, fusion_hidden=32, grad_checkpoint=False)
    cfg = TrainConfig(run_id="tiny", model=model, epochs=2, batch_studies=2, num_workers=0, amp_dtype="float32")
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def view_cfg(slots, view_crop_mm, **kw) -> TrainConfig:
    cfg = tiny_cfg(**kw)
    cfg.model.slots = tuple(slots)
    cfg.model.view_crop_mm = tuple(float(v) for v in view_crop_mm)
    return cfg


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_tiny_cache(tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="module")
def targets() -> pd.DataFrame:
    return synthetic_targets(STUDIES[:3], seed=1)


def test_sample_centres_uniform_and_random():
    c = sample_centres(30, 24, None)
    assert c.shape == (24,) and c.min() >= 0 and c.max() <= 29 and np.all(np.diff(c) >= 0)
    assert sample_centres(30, 24, None, z_offset=0.5).tolist() != c.tolist()
    assert sample_centres(3, 8, None).max() == 2  # fewer slices than K: repeats, never out of range
    rng = np.random.default_rng(0)
    r = sample_centres(32, 16, rng)
    assert r.shape == (16,) and np.all(np.diff(r) >= 0) and 0 <= r.min() and r.max() < 32
    assert all(2 * s <= v < 2 * s + 2 for s, v in enumerate(r))  # one index per stratum
    assert sample_centres(32, 16, np.random.default_rng(1)).tolist() != r.tolist()


def test_triplets_clamped_and_read_once(tiny_cache: Path):
    idx = triplet_indices(np.array([0, 5, 11]), 12)
    assert idx.tolist() == [[0, 0, 1], [4, 5, 6], [10, 11, 11]]
    vol = cache.load_series(tiny_cache, "train", "S1", "S1a")
    trip = read_triplets(vol, idx, 24)
    assert trip.shape == (3, 3, 24, 24) and trip.dtype == np.uint8
    expect = cv2.resize(np.asarray(vol[5]), (24, 24), interpolation=cv2.INTER_AREA)
    assert np.array_equal(trip[1, 1], expect)
    assert np.array_equal(trip[0, 0], trip[0, 1])


def test_warp_and_augment(tiny_cache: Path):
    vol = np.asarray(cache.load_series(tiny_cache, "train", "S1", "S1a"))
    plain = warp_slices(vol, PX, None)
    assert np.array_equal(plain, vol)
    cfg = tiny_cfg()
    p = augment_params(np.random.default_rng(0), cfg, 130.0)
    assert 1.0 <= p["scale"] <= 130.0 / 115.0 and abs(p["rot_deg"]) <= 8 and abs(p["contrast"] - 1) <= 0.03
    aug = augment_volume(vol, np.random.default_rng(0), cfg, px=32)
    assert aug.shape == (12, 32, 32) and aug.dtype == np.uint8
    assert not np.array_equal(aug, warp_slices(vol, 32, None))
    ident = warp_slices(vol, 32, {"scale": 1.0, "rot_deg": 0.0, "contrast": 1.0, "brightness": 0.0})
    assert np.abs(ident.astype(int) - warp_slices(vol, 32, None).astype(int)).mean() < 3
    bright = warp_slices(vol, 32, {"scale": 1.0, "rot_deg": 0.0, "contrast": 1.0, "brightness": 20.0})
    assert bright.astype(int).mean() > ident.astype(int).mean() + 10


def test_dataset_items_and_presence(tiny_cache: Path, targets: pd.DataFrame):
    cfg = tiny_cfg()
    ds = StudyDataset(tiny_cache, STUDIES, targets, cfg, train=False)
    assert len(ds) == 4
    it = ds[0]
    assert it["images"].shape == (3, 6, 3, 32, 32) and it["images"].dtype == torch.uint8
    assert it["present"].tolist() == [True, True, True]
    assert it["target"].shape == (12,) and it["weight"].shape == (12,)
    assert torch.equal(it["target"], torch.tensor(targets.iloc[0][[f"{k}_target" for k in LABEL_KEYS]].to_numpy(np.float32)))
    s2 = ds[1]
    assert s2["present"].tolist() == [True, False, True] and s2["images"][1].abs().sum() == 0
    s3 = ds[2]  # T1 fallbacks fill both FS slots
    assert s3["present"].tolist() == [True, True, True]
    s4 = ds[3]  # no targets row -> zero weights
    assert s4["weight"].sum() == 0 and s4["target"].sum() == 0
    a, b = ds[0]["images"], ds[0]["images"]
    assert torch.equal(a, b)  # inference sampling is deterministic
    tta = StudyDataset(tiny_cache, STUDIES, targets, cfg, train=False, z_offset=0.5)
    assert not torch.equal(tta[0]["images"], a)


def test_dataset_train_augmentation_and_seeding(tiny_cache: Path, targets: pd.DataFrame):
    cfg = tiny_cfg(sibling_p=0.0, slot_dropout_p=0.0)
    ds = StudyDataset(tiny_cache, STUDIES, targets, cfg, train=True, seed=3)
    it = ds[0]
    assert it["images"].shape == (3, 4, 3, 32, 32)
    assert torch.equal(it["images"], ds[0]["images"])  # same seed + epoch -> same sample
    ds.set_epoch(1)
    assert not torch.equal(it["images"], ds[0]["images"])
    ds2 = StudyDataset(tiny_cache, STUDIES, targets, cfg, train=True, seed=4)
    assert not torch.equal(it["images"], ds2[0]["images"])


def test_sibling_swap_and_slot_dropout(tiny_cache: Path, targets: pd.DataFrame):
    # S1 has a sagittal sibling with 8 instead of 12 slices; with sibling_p=1 it is always used.
    always = tiny_cfg(sibling_p=1.0, slot_dropout_p=0.0)
    never = tiny_cfg(sibling_p=0.0, slot_dropout_p=0.0)
    infer = StudyDataset(tiny_cache, ["S1"], targets, never, train=False)[0]["images"]
    diffs = 0
    for seed in range(6):
        a = StudyDataset(tiny_cache, ["S1"], targets, always, train=True, seed=seed)[0]["images"]
        b = StudyDataset(tiny_cache, ["S1"], targets, never, train=True, seed=seed)[0]["images"]
        diffs += int(not torch.equal(a[0], b[0]))
        assert a.shape == b.shape
    assert diffs == 6 and infer.shape[1] == 6
    drop = tiny_cfg(sibling_p=0.0, slot_dropout_p=1.0)
    for study, expect in (("S1", 2), ("S2", 2)):
        for seed in range(4):
            pres = StudyDataset(tiny_cache, [study], targets, drop, train=True, seed=seed)[0]["present"]
            assert int(pres.sum()) == expect, (study, seed, pres)


def test_collate_and_loader_speed(tiny_cache: Path, targets: pd.DataFrame):
    cfg = tiny_cfg()
    ds = StudyDataset(tiny_cache, STUDIES, targets, cfg, train=True)
    batch = collate_studies([ds[i] for i in range(4)])
    assert batch["images"].shape == (4, 3, 4, 3, 32, 32) and batch["present"].shape == (4, 3)
    assert batch["target"].shape == (4, 12) and batch["weight"].shape == (4, 12) and batch["study"] == STUDIES
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_studies, num_workers=0)
    t0 = time.monotonic()
    n = sum(b["images"].shape[0] for b in loader)
    assert n == 4 and time.monotonic() - t0 < 5.0


def test_slot_view_mm_defaults_and_validation():
    assert slot_view_mm(tiny_cfg()) == (130.0, 130.0, 130.0)
    four = ("sag_fs", "cor_fs", "ax_fs", "sag_fs")
    assert slot_view_mm(view_cfg(four, (130, 130, 130, 90))) == (130.0, 130.0, 130.0, 90.0)
    with pytest.raises(ValueError, match="entries for 3 slots"):
        slot_view_mm(view_cfg(("sag_fs", "cor_fs", "ax_fs"), (130, 90)))
    with pytest.raises(ValueError, match="must be in"):
        slot_view_mm(view_cfg(("sag_fs", "cor_fs", "ax_fs"), (130, 130, 200)))
    with pytest.raises(ValueError, match="duplicate"):
        slot_view_mm(view_cfg(("sag_fs", "sag_fs", "ax_fs"), (90, 90, 130)))
    with pytest.raises(ValueError, match="repeat"):
        slot_view_mm(view_cfg(("sag_fs", "sag_fs", "ax_fs"), ()))


def test_view_params_and_augment_scaling():
    assert view_params(130.0, 130.0) is None
    p = view_params(130.0, 90.0)
    assert p["scale"] == pytest.approx(130.0 / 90.0) and p["rot_deg"] == 0.0 and p["brightness"] == 0.0
    cfg = tiny_cfg()
    assert augment_params(np.random.default_rng(0), cfg, 130.0) == \
        augment_params(np.random.default_rng(0), cfg, 130.0, 130.0)
    for seed in range(8):
        s = augment_params(np.random.default_rng(seed), cfg, 130.0, 90.0)["scale"]
        assert 130.0 / 90.0 <= s <= 130.0 / (90.0 * 115.0 / 130.0)


def test_extra_view_crops_the_same_series(tiny_cache: Path, targets: pd.DataFrame, monkeypatch):
    cfg = view_cfg(("sag_fs", "cor_fs", "ax_fs", "sag_fs"), (130, 130, 130, 90))
    loaded: list[str] = []
    orig = cache.load_series
    monkeypatch.setattr(cache, "load_series", lambda *a, **kw: (loaded.append(a[3]), orig(*a, **kw))[1])
    it = StudyDataset(tiny_cache, ["S1"], targets, cfg, train=False)[0]
    assert it["images"].shape == (4, 6, 3, 32, 32) and it["present"].tolist() == [True] * 4
    assert loaded == ["S1a", "S1c", "S1d"]  # the repeated slot is read once and cropped twice
    vol = np.asarray(orig(tiny_cache, "train", "S1", "S1a"))
    idx = triplet_indices(sample_centres(vol.shape[0], 6, None), vol.shape[0])
    assert np.array_equal(it["images"][3].numpy(), read_triplets(vol, idx, 32, view_params(130.0, 90.0)))
    assert np.array_equal(it["images"][0].numpy(), read_triplets(vol, idx, 32, None))


def test_view_crop_keeps_the_default_path_identical(tiny_cache: Path, targets: pd.DataFrame):
    slots = ("sag_fs", "cor_fs", "ax_fs")
    for train in (False, True):
        a = StudyDataset(tiny_cache, STUDIES, targets, tiny_cfg(), train=train, seed=5)
        b = StudyDataset(tiny_cache, STUDIES, targets, view_cfg(slots, (130, 130, 130)), train=train, seed=5)
        for i in range(len(STUDIES)):
            assert torch.equal(a[i]["images"], b[i]["images"]) and torch.equal(a[i]["present"], b[i]["present"])


def test_targets_io(tmp_path: Path):
    df = synthetic_targets(["a", "b"], seed=0)
    assert len(df) == 2 and set(df.columns) == {ID_COL, *[f"{k}_target" for k in LABEL_KEYS],
                                                 *[f"{k}_weight" for k in LABEL_KEYS]}
    df.to_parquet(tmp_path / "t.parquet", index=False)
    back = load_targets(tmp_path / "t.parquet")
    pd.testing.assert_frame_equal(back, df)
    df.drop(columns=["acl_weight"]).to_parquet(tmp_path / "bad.parquet", index=False)
    with pytest.raises(ValueError, match="acl_weight"):
        load_targets(tmp_path / "bad.parquet")
