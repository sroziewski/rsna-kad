import numpy as np
import pytest

from conftest import DISC_MM, OFFSET_DISC_MM, UID_ROOT, SyntheticData
from rsna_kad import dicom, preprocess
from rsna_kad.config import CacheConfig

CFG = CacheConfig()
MM_PER_PX = CFG.crop_mm / CFG.px


def _load(synthetic: SyntheticData, uid: str):
    spec = synthetic.spec(uid)
    headers = [dicom.read_header(p) for p in dicom.sort_series(spec.paths)]
    geom = dicom.series_geometry(headers)
    return [dicom.read_slice(h.path) for h in headers], geom


def _disc_radius_px(img: np.ndarray, thr: float) -> float:
    return float(np.sqrt((img > thr).sum() / np.pi))


def test_fill_missing_uses_neighbour_mean():
    a, c = np.full((4, 4), 10.0, np.float32), np.full((4, 4), 30.0, np.float32)
    filled, n = preprocess.fill_missing([a, None, c, None])
    assert n == 2
    assert np.allclose(filled[1], 20.0)
    assert np.allclose(filled[3], 30.0)
    with pytest.raises(ValueError):
        preprocess.fill_missing([None, None])


def test_crop_205mm_fov_gives_correct_mm_per_px(synthetic: SyntheticData):
    slices, geom = _load(synthetic, UID_ROOT + "200001")
    img = slices[15]
    out = preprocess.crop_resample(img, geom.spacing, CFG.crop_mm, CFG.px)
    assert out.shape == (384, 384) and out.dtype == np.float32
    thr = 0.3 * img.max()
    assert _disc_radius_px(out, thr) == pytest.approx(DISC_MM / MM_PER_PX, abs=3)
    # The offset disc sits +30 mm along the row direction, i.e. to the right of centre.
    cy, cx = 191.5, 191.5
    ys, xs = np.nonzero(out > 0.95 * img.max())
    assert abs(ys.mean() - cy) < 3
    assert xs.mean() - cx == pytest.approx(OFFSET_DISC_MM / MM_PER_PX, abs=3)


def test_crop_120mm_fov_pads_and_keeps_scale(synthetic: SyntheticData):
    slices, geom = _load(synthetic, UID_ROOT + "200002")
    assert geom.fov_mm < CFG.crop_mm
    out = preprocess.crop_resample(slices[60], geom.spacing, CFG.crop_mm, CFG.px)
    assert out.shape == (384, 384)
    # zero padding ring outside the 120 mm FOV
    pad_px = int((CFG.crop_mm - geom.fov_mm) / 2 / MM_PER_PX) - 2
    assert pad_px > 5
    assert out[:pad_px].max() == 0 and out[:, :pad_px].max() == 0
    assert _disc_radius_px(out, 0.3 * slices[60].max()) == pytest.approx(DISC_MM / MM_PER_PX, abs=3)


def test_crop_non_square(synthetic: SyntheticData):
    slices, geom = _load(synthetic, UID_ROOT + "200003")
    assert (geom.rows, geom.cols) == (640, 1280)
    out = preprocess.crop_resample(np.stack(slices[:3]), geom.spacing, CFG.crop_mm, CFG.px)
    assert out.shape == (3, 384, 384)
    assert _disc_radius_px(out[1], 0.3 * slices[1].max()) == pytest.approx(DISC_MM / MM_PER_PX, abs=3)


def test_resample_z_averages_slabs():
    vol = np.arange(10, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
    out, step = preprocess.resample_z(vol, 0.9, 3.0)
    assert out.shape == (4, 2, 2) and step == pytest.approx(2.7)
    assert np.allclose(out[:, 0, 0], [1.0, 4.0, 7.0, 9.0])
    same, step2 = preprocess.resample_z(vol, 3.0, 3.0)
    assert same is vol and step2 == 3.0
    same3, _ = preprocess.resample_z(vol, 2.5, 3.0)
    assert same3 is vol


def test_subsample_indices():
    idx = preprocess.subsample_indices(64, 40)
    assert idx.size == 40 and idx[0] == 0 and idx[-1] == 63 and np.all(np.diff(idx) > 0)
    assert np.array_equal(preprocess.subsample_indices(30, 40), np.arange(30))


def test_robust_window():
    vol = np.linspace(-100, 900, 10001, dtype=np.float32).reshape(1, 1, -1)
    u8, lo, hi = preprocess.robust_window(vol, 1.0, 99.0)
    assert u8.dtype == np.uint8 and u8.min() == 0 and u8.max() == 255
    assert lo == pytest.approx(-90, abs=1) and hi == pytest.approx(890, abs=1)
    flat, lo2, hi2 = preprocess.robust_window(np.zeros((1, 4, 4), np.float32), 1, 99)
    assert flat.max() == 0 and hi2 > lo2


def test_3d_series_resampled_to_3mm_and_capped(synthetic: SyntheticData):
    slices, geom = _load(synthetic, UID_ROOT + "200002")
    vol, meta = preprocess.series_to_volume(slices, geom, CFG)
    assert vol.shape == (40, 384, 384) and vol.dtype == np.uint8
    assert meta.n_src == 120 and meta.n_cached == 40 and meta.truncated_slices == 0
    assert meta.is_3d and meta.z_step_cached_mm == pytest.approx(2.7)
    assert 20 <= meta.mean_u8 <= 200


def test_2d_series_with_truncated_slice(synthetic: SyntheticData):
    slices, geom = _load(synthetic, UID_ROOT + "200001")
    assert slices[10] is None
    vol, meta = preprocess.series_to_volume(slices, geom, CFG)
    assert vol.shape == (30, 384, 384)
    assert meta.truncated_slices == 1 and meta.n_src == 30 and meta.n_cached == 30
    assert not meta.is_3d and meta.z_step_cached_mm == pytest.approx(3.0)
    # the filled slice is the neighbour mean, so it sits between its neighbours
    mid = vol[10].astype(np.float32)
    assert np.abs(mid - (vol[9].astype(np.float32) + vol[11]) / 2).mean() < 3.0
    assert 0 < meta.p_lo < meta.p_hi


def test_long_2d_series_subsampled_to_cap(synthetic: SyntheticData):
    slices, geom = _load(synthetic, UID_ROOT + "200004")
    many = slices * 2  # 50 slices, still tagged 2D
    from dataclasses import replace
    vol, meta = preprocess.series_to_volume(many, replace(geom, n_slices=50), CFG)
    assert vol.shape[0] == 40 and meta.n_src == 50
    assert meta.p_lo < 0 < meta.p_hi  # signed input windowed without clipping the negatives away
