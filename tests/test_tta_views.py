"""Evaluation views for TTA beyond z-TTA (rsna_kad.tta_views): identity reproduces the plain eval
dataset byte-for-byte, the crop view shrinks every field by the same factor, rotation uses the
training warp, and nothing flips."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rsna_kad.data import StudyDataset, view_params, warp_slices
from rsna_kad.tta_views import ViewTTADataset, eval_view_params
from test_data import make_tiny_cache, tiny_cfg


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory):
    return make_tiny_cache(tmp_path_factory.mktemp("tta_views") / "cache")


def _ds(cache, cfg, cls=StudyDataset, **kw):
    return cls(cache, ["S1", "S2", "S3"], None, cfg, train=False, **kw)


def test_identity_view_equals_plain_eval_dataset(cache):
    cfg = tiny_cfg()
    a, b = _ds(cache, cfg), _ds(cache, cfg, ViewTTADataset)
    for i in range(len(a)):
        x, y = a[i], b[i]
        assert torch.equal(x["images"], y["images"]) and torch.equal(x["present"], y["present"])


def test_eval_view_params_matches_view_params_and_scales_every_field():
    for crop, view in ((130.0, 130.0), (130.0, 90.0)):
        assert eval_view_params(crop, view) == view_params(crop, view)
    f = 115.0 / 130.0
    assert eval_view_params(130.0, 130.0, f)["scale"] == pytest.approx(130.0 / 115.0)
    assert eval_view_params(130.0, 90.0, f)["scale"] == pytest.approx(130.0 / (90.0 * f))
    p = eval_view_params(130.0, 130.0, 1.0, 5.0)
    assert p == {"scale": 1.0, "rot_deg": 5.0, "contrast": 1.0, "brightness": 0.0}


def test_crop_and_rotation_change_the_input_but_not_the_layout(cache):
    cfg = tiny_cfg()
    base = _ds(cache, cfg)[0]["images"]
    for kw in ({"crop_factor": 115 / 130}, {"rot_deg": 5.0}, {"rot_deg": -5.0}):
        img = _ds(cache, cfg, ViewTTADataset, **kw)[0]["images"]
        assert img.shape == base.shape and img.dtype == base.dtype and not torch.equal(img, base)


def test_rotation_is_the_training_warp_and_not_a_flip():
    rng = np.random.default_rng(0)
    sl = rng.integers(0, 255, (2, 64, 64), dtype=np.uint8)
    sl[:, :, :8] = 255                                            # a bright left edge
    p = eval_view_params(64.0, 64.0, 1.0, 5.0)
    out = warp_slices(sl, 32, p)
    assert np.array_equal(out, warp_slices(sl, 32, dict(p)))
    assert out[:, 8:24, :3].mean() > out[:, 8:24, -3:].mean()    # the bright side stays on the left


def test_training_mode_and_bad_factor_are_rejected(cache):
    cfg = tiny_cfg()
    with pytest.raises(ValueError):
        ViewTTADataset(cache, ["S1"], None, cfg, train=True)
    with pytest.raises(ValueError):
        ViewTTADataset(cache, ["S1"], None, cfg, train=False, crop_factor=1.2)
