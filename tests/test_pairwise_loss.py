"""Optional pairwise ranking term (TrainConfig.loss = "bce_pairwise"): the default path is
unchanged, weight 0 reproduces BCE exactly, the pair rule is metrics.binarise's, and the per-label
bank gives rare labels pairs that a 4-study batch almost never holds."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from rsna_kad import train as tr
from rsna_kad.config import TrainConfig
from rsna_kad.metrics import binarise
from test_train import env, patch_dirs  # noqa: F401  (fixtures)

L = 3


def _batch(target, weight=None, logits=None):
    t = torch.tensor(target, dtype=torch.float32)
    w = torch.ones_like(t) if weight is None else torch.tensor(weight, dtype=torch.float32)
    z = torch.zeros_like(t) if logits is None else torch.tensor(logits, dtype=torch.float32)
    return z.requires_grad_(True), t, w


def test_defaults():
    cfg = TrainConfig()
    assert cfg.loss == "bce" and cfg.pairwise_weight == 0.12


def test_pair_rule_is_metrics_binarise():
    t = np.array([[0.6], [0.2], [0.2], [0.59], [0.9]], dtype=np.float32)
    w = np.array([[1.0], [0.5], [0.4], [1.0], [0.1]], dtype=np.float32)
    pos, neg = tr.PairwiseRankLoss._masks(torch.from_numpy(t), torch.from_numpy(w))
    keep, y = binarise(t[:, 0], w[:, 0])
    assert pos[:, 0].tolist() == list(keep & y) and neg[:, 0].tolist() == list(keep & ~y)


def test_large_margin_gives_zero_and_no_pairs_gives_exact_zero():
    z, t, w = _batch([[1, 0, 0.4], [0, 1, 0.4], [1, 0, 0.4]], logits=[[50, -50, 0], [-50, 50, 0], [50, -50, 0]])
    loss = tr.PairwiseRankLoss(L)(z, t, w)
    assert 0.0 <= float(loss) < 1e-20
    z, t, w = _batch([[0.4] * L, [0.4] * L])                       # middle band only: no pair anywhere
    loss = tr.PairwiseRankLoss(L)(z, t, w)
    assert float(loss) == 0.0
    loss.backward()
    assert torch.equal(z.grad, torch.zeros_like(z))


def test_misranked_pairs_have_finite_gradient_in_the_right_direction():
    rng = np.random.default_rng(0)
    z, t, w = _batch(rng.integers(0, 2, (4, L)).astype(float), logits=rng.normal(0, 3, (4, L)))
    t[:, 0] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    with torch.no_grad():
        z[:, 0] = torch.tensor([-2.0, 2.0, -1.0, 1.0])            # every positive below every negative
    loss = tr.PairwiseRankLoss(L)(z, t, w)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert (z.grad[[0, 2], 0] < 0).all() and (z.grad[[1, 3], 0] > 0).all()   # descent raises pos, lowers neg


def test_bank_supplies_pairs_for_a_label_without_in_batch_negatives():
    rank = tr.PairwiseRankLoss(1, bank=4)
    for _ in range(3):                                           # negatives only, empty positive bank: no pair
        z, t, w = _batch([[0.0], [0.0]], logits=[[1.0], [2.0]])
        assert float(rank(z, t, w)) == 0.0
    assert rank.neg[0].numel() == 4 and rank.pos[0] is None       # bank capped at 4, detached
    z, t, w = _batch([[1.0]], logits=[[0.0]])                     # a lone positive
    loss = rank(z, t, w)
    loss.backward()
    assert float(loss) > 0.0 and float(z.grad[0, 0]) < 0.0
    assert not rank.neg[0].requires_grad


def test_unknown_loss_is_rejected(env, tmp_path: Path):  # noqa: F811
    cfg = copy.deepcopy(env["cfg"])
    cfg.loss = "svm"
    with pytest.raises(ValueError, match="loss must be one of"):
        tr.train_fold(cfg, 0, tmp_path / "r" / "fold0", device="cpu", pretrained=False)


def test_weight_zero_reproduces_the_default_bce_run_exactly(env, tmp_path: Path):  # noqa: F811
    base = copy.deepcopy(env["cfg"])
    pw0 = copy.deepcopy(env["cfg"])
    pw0.loss, pw0.pairwise_weight = "bce_pairwise", 0.0
    pw = copy.deepcopy(env["cfg"])
    pw.loss = "bce_pairwise"
    out = {}
    for name, cfg in (("bce", base), ("pw0", pw0), ("pw", pw)):
        rd = tmp_path / name / "fold0"
        tr.train_fold(cfg, 0, rd, device="cpu", pretrained=False)
        out[name] = load_file(str(rd / "model.safetensors"))
    assert out["bce"].keys() == out["pw0"].keys()
    assert all(torch.equal(out["bce"][k], out["pw0"][k]) for k in out["bce"])
    assert any(not torch.equal(out["bce"][k], out["pw"][k]) for k in out["bce"])
