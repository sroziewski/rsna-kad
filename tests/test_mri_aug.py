"""MRI intensity augmentation (TrainConfig.mri_aug, rsna_kad.mri_aug): off by default and then never
called, deterministic under a seed, one parameter set per series, training only, uint8 range kept."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_kad import mri_aug
from rsna_kad.config import TrainConfig
from rsna_kad.data import StudyDataset
from test_data import make_tiny_cache, tiny_cfg


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory):
    return make_tiny_cache(tmp_path_factory.mktemp("mri_aug") / "cache")


def _series(seed: int = 0, n: int = 4, px: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    one = rng.integers(0, 256, (px, px), dtype=np.uint8)
    return np.repeat(one[None, None], n, axis=0).repeat(3, axis=1)          # (n, 3, px, px), identical slices


ALL_ON = {"bias": np.full(mri_aug.n_bias_coefs(), 0.3), "gamma": 1.35, "blur": 0.8, "noise": 0.03}


def test_default_is_off():
    assert TrainConfig().mri_aug is False


def test_seed_makes_it_deterministic():
    x = _series()
    a = mri_aug.mri_augment(x, np.random.default_rng(7))
    b = mri_aug.mri_augment(x, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_one_parameter_set_for_every_slice_of_a_series():
    x = _series()
    no_noise = {k: v for k, v in ALL_ON.items() if k != "noise"}
    y = mri_aug.apply_params(x, no_noise, np.random.default_rng(0))
    assert not np.array_equal(y, x)
    assert all(np.array_equal(y[0, 0], y[i, c]) for i in range(x.shape[0]) for c in range(3))
    # with noise the noise differs per pixel, but its scale is the series' one
    z = mri_aug.apply_params(x, ALL_ON, np.random.default_rng(0)).astype(float)
    d = (z - mri_aug.apply_params(x, no_noise, np.random.default_rng(0)).astype(float)) / 255.0
    assert abs(d.std() - 0.03) < 0.01


@pytest.mark.parametrize("value", [0, 128, 255])
def test_value_range_dtype_and_shape_are_kept(value: int):
    x = np.full((3, 3, 32, 32), value, np.uint8)
    for p in (ALL_ON, {"bias": -np.full(mri_aug.n_bias_coefs(), 0.3)}, {"gamma": 0.74}):
        y = mri_aug.apply_params(x, p, np.random.default_rng(1))
        assert y.dtype == np.uint8 and y.shape == x.shape


def test_intensity_only_no_pixel_moves():
    x = np.zeros((2, 3, 32, 32), np.uint8)
    x[..., 4:10, 2:8] = 200                                             # a block on the left
    y = mri_aug.apply_params(x, {"gamma": 1.3, "bias": np.full(mri_aug.n_bias_coefs(), 0.1)}, np.random.default_rng(0))
    assert y[..., 4:10, 2:8].min() > 0 and y[..., :, 16:].max() == 0


def test_bias_field_is_positive_and_smooth():
    f = mri_aug.bias_field(32, 32, np.random.default_rng(0).uniform(-0.3, 0.3, mri_aug.n_bias_coefs()))
    assert mri_aug.n_bias_coefs() == 10 and (f > 0).all() and np.abs(np.diff(f, axis=0)).max() < 0.2


def test_dataset_off_never_calls_it_and_eval_never_does(cache, monkeypatch: pytest.MonkeyPatch):
    def boom(*a, **k):
        raise AssertionError("mri_augment called")
    cfg = tiny_cfg()
    ref = [StudyDataset(cache, ["S1", "S2"], None, cfg, train=True, seed=3)[i]["images"] for i in range(2)]
    monkeypatch.setattr(mri_aug, "mri_augment", boom)
    for i in range(2):                                                   # default off: same bytes, never called
        assert torch.equal(StudyDataset(cache, ["S1", "S2"], None, cfg, train=True, seed=3)[i]["images"], ref[i])
    on = tiny_cfg(mri_aug=True)
    ev_on = StudyDataset(cache, ["S1", "S2"], None, on, train=False)[0]["images"]
    ev_off = StudyDataset(cache, ["S1", "S2"], None, cfg, train=False)[0]["images"]
    assert torch.equal(ev_on, ev_off)                                    # eval ignores it


def test_dataset_on_augments_once_per_present_series(cache, monkeypatch: pytest.MonkeyPatch):
    calls = []
    real = mri_aug.mri_augment

    def spy(img, rng):
        calls.append(img.shape)
        return real(img, rng)
    monkeypatch.setattr(mri_aug, "mri_augment", spy)
    cfg = tiny_cfg(mri_aug=True, slot_dropout_p=0.0)
    ds = StudyDataset(cache, ["S1"], None, cfg, train=True, seed=3)
    item = ds[0]
    assert len(calls) == int(item["present"].sum()) and all(len(s) == 4 for s in calls)
    again = StudyDataset(cache, ["S1"], None, cfg, train=True, seed=3)[0]["images"]
    assert torch.equal(item["images"], again)                            # seeded: deterministic
