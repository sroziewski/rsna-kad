"""KneeMIL forward/fusion contract and the safetensors bundle (§2.4, §3.2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from rsna_kad.config import ModelConfig, TrainConfig, load_config
from rsna_kad.constants import LABEL_KEYS, LABELS, MAIN_SLOTS
from rsna_kad.model import (BUNDLE_NAME, HEAD_LABEL, HEAD_RESIDUAL, HEAD_SHARED, HEAD_XFMR, WEIGHTS_NAME, KneeMIL,
                            NARROW_VIEW_KEYS, SLOT_PRIOR, build_model, bundle_meta, fingerprint,
                            load_bundle, masked_bce, normalize, prior_matrix, save_bundle)


def small_cfg(slots=("sag_fs", "cor_fs", "ax_fs"), **kw) -> ModelConfig:
    return ModelConfig(backbone="resnet18", img_px=32, slots=slots, k_train=4, k_infer=4, attn_hidden=16,
                       fusion_hidden=32, grad_checkpoint=False, label_attn_hidden=16,
                       **{"xfmr_dim": 32, "xfmr_heads": 4, "xfmr_layers": 2, **kw})


def batch(n_slots: int, k: int = 4, px: int = 32, b: int = 2):
    torch.manual_seed(0)
    images = torch.randint(0, 256, (b, n_slots, k, 3, px, px), dtype=torch.uint8)
    present = torch.ones(b, n_slots, dtype=torch.bool)
    present[1, 1] = False
    return images, present


def test_forward_two_studies_one_missing_slot():
    for slots in (("sag_fs", "cor_fs", "ax_fs"), MAIN_SLOTS):
        model = build_model(small_cfg(slots), pretrained=False).eval()
        images, present = batch(len(slots))
        with torch.no_grad():
            logits, aux = model(images, present)
        assert logits.shape == (2, 12) and aux.shape == (2, len(slots), 12)
        assert torch.isfinite(logits).all() and torch.isfinite(aux).all()
        emb = model.embed(images, present)
        assert emb.shape == (2, len(slots), model.dim) and emb[1, 1].abs().sum() == 0
        filled = model.fill(emb, present.float())
        assert torch.equal(filled[1, 1], model.missing[1]) and torch.equal(filled[0, 1], emb[0, 1])


def test_missing_slot_changes_prediction_and_all_missing_is_finite():
    model = build_model(small_cfg(), pretrained=False).eval()
    images, present = batch(3)
    with torch.no_grad():
        a, _ = model(images, present)
        b, _ = model(images, torch.ones_like(present))
        c, aux = model(images, torch.zeros_like(present))
    assert not torch.allclose(a[1], b[1], atol=1e-4) and torch.allclose(a[0], b[0], atol=1e-4)
    assert torch.isfinite(c).all() and torch.isfinite(aux).all()


def test_backward_and_mixup_path():
    model = build_model(small_cfg(), pretrained=False).train()
    images, present = batch(3)
    target = torch.rand(2, 12)
    weight = torch.rand(2, 12)
    pf = present.float()
    filled = model.fill(model.embed(images, present), pf)
    lam = 0.3
    mixed = lam * filled + (1 - lam) * filled.flip(0)
    pf_mixed = lam * pf + (1 - lam) * pf.flip(0)
    logits, aux = model.fuse(mixed, pf_mixed)
    loss = masked_bce(logits, target, weight) + 0.3 * masked_bce(
        aux, target.unsqueeze(1).expand_as(aux), weight.unsqueeze(1) * pf_mixed.unsqueeze(2))
    loss.backward()
    assert torch.isfinite(loss) and model.missing.grad is not None and model.missing.grad.abs().sum() > 0
    assert any(p.grad is not None for p in model.backbone.parameters())
    heads = model.head_parameters()
    n_all = sum(p.numel() for p in model.parameters())
    assert n_all == sum(p.numel() for p in heads) + sum(p.numel() for p in model.backbone.parameters())


def test_masked_bce_and_normalize():
    logits = torch.tensor([[0.0, 10.0]])
    target = torch.tensor([[1.0, 0.0]])
    assert masked_bce(logits, target, torch.tensor([[1.0, 0.0]])).item() == pytest.approx(np.log(2), abs=1e-5)
    assert masked_bce(logits, target, torch.tensor([[0.0, 1.0]])).item() == pytest.approx(10.0, abs=1e-3)
    x = torch.full((1, 3, 2, 2), 255, dtype=torch.uint8)
    y = normalize(x)
    assert y.dtype == torch.float32 and y[0, 0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229, abs=1e-5)
    assert normalize(y) is y


def test_bundle_roundtrip_bit_exact(tmp_path: Path):
    cfg = TrainConfig(run_id="t", model=small_cfg())
    model = build_model(cfg.model, pretrained=False)
    meta = bundle_meta(cfg, fold=0, metrics={"oof_weak_auc": 0.5})
    doc = save_bundle(model, meta, tmp_path)
    assert (tmp_path / WEIGHTS_NAME).exists() and (tmp_path / BUNDLE_NAME).exists()
    saved = json.loads((tmp_path / BUNDLE_NAME).read_text())
    assert saved["arch"]["backbone"] == "resnet18" and saved["arch"]["slots"] == ["sag_fs", "cor_fs", "ax_fs"]
    assert saved["labels"] == list(LABELS) and len(saved["fingerprint"]) == 8 and saved["fold"] == 0
    assert saved["config"]["model"]["img_px"] == 32 and saved["metrics"]["oof_weak_auc"] == 0.5
    assert doc["fingerprint"] == saved["fingerprint"]
    back, meta2 = load_bundle(tmp_path, "cpu")
    assert meta2 == saved and isinstance(back, KneeMIL) and not back.training
    for (k, v), (k2, v2) in zip(model.state_dict().items(), back.state_dict().items()):
        assert k == k2 and torch.equal(v, v2), k
    images, present = batch(3)
    with torch.no_grad():
        assert torch.equal(model.eval()(images, present)[0], back(images, present)[0])
    assert fingerprint(back.state_dict()) == saved["fingerprint"]


def test_bundle_with_view_crops_needs_a_view_aware_caller(tmp_path: Path):
    mcfg = small_cfg(slots=("sag_fs", "cor_fs", "ax_fs", "sag_fs"))
    mcfg.view_crop_mm = (130.0, 130.0, 130.0, 90.0)
    cfg = TrainConfig(run_id="t", model=mcfg)
    save_bundle(build_model(mcfg, pretrained=False), bundle_meta(cfg, 0), tmp_path)
    assert json.loads((tmp_path / BUNDLE_NAME).read_text())["arch"]["view_crop_mm"] == [130.0, 130.0, 130.0, 90.0]
    with pytest.raises(RuntimeError, match="view crops"):
        load_bundle(tmp_path, "cpu")
    model, _ = load_bundle(tmp_path, "cpu", views=True)
    assert model.slots == ("sag_fs", "cor_fs", "ax_fs", "sag_fs")


# ------------------------------------------------------------------ head option (§4 A2)

# Forward of the pre-A2 implementation (resnet18 @32, MAIN_SLOTS, K=4, seed 0, one missing slot),
# recorded on the server before the head option existed. `shared_attn` must not move by a bit:
# arms A/A8/C and every exported bundle were trained with exactly this arithmetic.
PINNED_SHARED_LOGITS = [
    -0.155044, 0.200538, 0.03897, -0.007903, 0.11214, -0.070065, 0.086538, 0.181645, 0.039488,
    0.051593, -0.0417, -0.003537, -0.154692, 0.187171, 0.023226, -0.001044, 0.11049, -0.0803,
    0.081468, 0.189738, 0.047897, 0.074565, -0.029616, -0.00428]
PINNED_SHARED_AUX_SUM = -1.804344
PINNED_SHARED_N_PARAM = 11285709
PINNED_SHARED_N_STATE = 133


def pinned_model():
    torch.manual_seed(0)
    model = build_model(small_cfg(MAIN_SLOTS), pretrained=False).eval()
    torch.manual_seed(0)
    images = torch.randint(0, 256, (2, 4, 4, 3, 32, 32), dtype=torch.uint8)
    present = torch.ones(2, 4, dtype=torch.bool)
    present[1, 1] = False
    return model, images, present


def test_shared_attn_is_bit_identical_to_the_pre_a2_head():
    model, images, present = pinned_model()
    assert model.cfg.head == HEAD_SHARED and not hasattr(model, "label_head")
    assert sum(p.numel() for p in model.parameters()) == PINNED_SHARED_N_PARAM
    assert len(model.state_dict()) == PINNED_SHARED_N_STATE
    with torch.no_grad():
        logits, aux = model(images, present)
    assert logits.flatten().tolist() == pytest.approx(PINNED_SHARED_LOGITS, abs=2e-5)
    assert float(aux.double().sum()) == pytest.approx(PINNED_SHARED_AUX_SUM, abs=2e-5)


def test_shared_attn_matches_an_independent_recomputation_of_2_4():
    """The pinned numbers above say 'unchanged'; this says 'unchanged AND still §2.4'."""
    model, images, present = pinned_model()
    with torch.no_grad():
        logits, aux = model(images, present)
        pooled = torch.zeros(2, 4, model.dim)
        for i in range(2):
            for j in range(4):
                if not present[i, j]:
                    continue
                h = model.backbone(normalize(images[i, j]))
                g = torch.tanh(model.attn.v(h)) * torch.sigmoid(model.attn.u(h))
                pooled[i, j] = model.attn.w(g).squeeze(-1).softmax(0) @ h
        pf = present.float()
        filled = pooled * pf.unsqueeze(-1) + model.missing * (1.0 - pf.unsqueeze(-1))
        ref = model.fusion(torch.cat([filled.flatten(1), pf], dim=1))
        ref_aux = torch.einsum("bsd,sdl->bsl", filled, model.aux_w) + model.aux_b
    assert torch.allclose(logits, ref, atol=2e-4) and torch.allclose(aux, ref_aux, atol=2e-4)


def test_existing_configs_keep_the_shared_head():
    for name in ("main_convnext_s_384.yaml", "main_convnext_s_384_e8.yaml",
                 "main_armc_convnextv2_t_384.yaml", "eff_effv2s_256.yaml", "mvp_effv2s_256.yaml"):
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / name)
        assert cfg.model.head == HEAD_SHARED, name


def test_unknown_head_is_refused():
    with pytest.raises(ValueError, match="model.head"):
        build_model(small_cfg(head="attention"), pretrained=False)


@pytest.mark.parametrize("head", [HEAD_LABEL, HEAD_RESIDUAL, HEAD_XFMR])
def test_per_label_head_shapes_and_finiteness_with_a_missing_slot(head):
    for slots in (("sag_fs", "cor_fs", "ax_fs"), MAIN_SLOTS):
        model = build_model(small_cfg(slots, head=head), pretrained=False).eval()
        images, present = batch(len(slots))
        with torch.no_grad():
            logits, aux = model(images, present)
            none, none_aux = model(images, torch.zeros_like(present))
        assert logits.shape == (2, 12) and aux.shape == (2, len(slots), 12)
        assert torch.isfinite(logits).all() and torch.isfinite(aux).all()
        assert torch.isfinite(none).all() and torch.isfinite(none_aux).all()
        bag = model.embed(images, present)
        assert bag.shape == (2, len(slots) * (1 + model.cfg.k_infer), model.dim)
        # The [MISSING] vector still stands in for the absent slot, and its triplet tokens are 0.
        filled = model.fill(bag, present.float())
        assert torch.equal(filled[1, 1], model.missing[1])
        k = model.cfg.k_infer
        assert filled[1, len(slots) + 1 * k:len(slots) + 2 * k].abs().sum() == 0


@pytest.mark.parametrize("head", [HEAD_LABEL, HEAD_RESIDUAL, HEAD_XFMR])
def test_label_attention_is_a_distribution_per_label_and_masks_absent_slots(head):
    model = build_model(small_cfg(MAIN_SLOTS, head=head), pretrained=False).eval()
    images, present = batch(4)
    with torch.no_grad():
        a = model.label_attention(images, present)
    s, k = 4, model.cfg.k_infer
    assert a.shape == (2, s + s * k, 12)
    assert torch.allclose(a.sum(dim=1), torch.ones(2, 12), atol=1e-5)
    # Study 1 has slot 1 absent: every triplet token of that slot carries ~no mass for any label
    # (the slot summary token, which holds the [MISSING] vector, is deliberately not masked).
    absent = a[1, s + 1 * k:s + 2 * k]
    assert float(absent.max()) < 1e-5 and float(a[0, s + k:s + 2 * k].sum()) > 1e-3


def test_residual_head_with_gate_zero_is_exactly_the_baseline_pooling():
    model = build_model(small_cfg(MAIN_SLOTS, head=HEAD_RESIDUAL), pretrained=False).eval()
    images, present = batch(4)
    pf = present.float()
    with torch.no_grad():
        assert float(model.gate.abs().sum()) == 0.0
        filled = model.fill(model.embed(images, present), pf)
        base = model.fusion(torch.cat([filled[:, :4].flatten(1), pf], dim=1))
        logits, _ = model.fuse(filled, pf)
        assert torch.equal(logits, base)
        model.gate.fill_(1.0)
        moved, _ = model.fuse(filled, pf)
    assert not torch.allclose(moved, base, atol=1e-6)


@pytest.mark.parametrize("head", [HEAD_LABEL, HEAD_RESIDUAL, HEAD_XFMR])
@pytest.mark.parametrize("gate", [0.0, 0.5])
def test_per_label_head_trains_end_to_end(head, gate):
    model = build_model(small_cfg(MAIN_SLOTS, head=head), pretrained=False).train()
    if head == HEAD_RESIDUAL:
        with torch.no_grad():
            model.gate.fill_(gate)
    images, present = batch(4)
    pf = present.float()
    filled = model.fill(model.embed(images, present), pf)
    # train.train_fold's manifold mixup, applied to whatever `fill` returns.
    mixed = 0.3 * filled + 0.7 * filled.flip(0)
    pf_mixed = 0.3 * pf + 0.7 * pf.flip(0)
    logits, aux = model.fuse(mixed, pf_mixed)
    loss = masked_bce(logits, torch.rand(2, 12), torch.rand(2, 12)) + 0.3 * masked_bce(
        aux, torch.rand(2, 12).unsqueeze(1).expand_as(aux), torch.rand(2, 4, 12))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits).all()
    assert model.missing.grad is not None and model.missing.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    live = head != HEAD_RESIDUAL or gate != 0.0
    g = model.label_head.cls_w.grad
    assert bool(g is not None and g.abs().sum() > 0) is live
    if head == HEAD_RESIDUAL:
        # At gate 0 the delta path is gradient-starved by construction - the loss reaches the
        # gate itself, which moves off zero on the first step and opens the path. That is the
        # whole point of the conservative variant, so pin it rather than initialise around it.
        assert model.gate.grad is not None and model.gate.grad.abs().sum() > 0


def test_prior_matrix_table():
    p = prior_matrix(MAIN_SLOTS, (), 130.0, 0.55)
    assert p.shape == (12, 4)
    row = {k: i for i, k in enumerate(LABEL_KEYS)}
    assert p[row["mcl"]].tolist() == pytest.approx([0.0, 0.55, 0.0, 0.0])      # coronal only
    assert p[row["acl"]].tolist() == pytest.approx([0.55, 0.0, 0.0, 0.55])     # sagittal only
    assert p[row["fracture"]].tolist() == pytest.approx([0.55] * 4)            # uniform = no-op
    assert prior_matrix(MAIN_SLOTS, (), 130.0, 0.0).abs().sum() == 0.0
    # arm C's 5th view is a 90 mm centre crop of sag_fs: only central findings keep a prior there.
    five = MAIN_SLOTS + ("sag_fs",)
    q = prior_matrix(five, (130.0, 130.0, 130.0, 130.0, 90.0), 130.0, 0.55)
    assert q.shape == (12, 5)
    assert torch.equal(q[:, :4], p)
    for i, k in enumerate(LABEL_KEYS):
        assert float(q[i, 4]) == pytest.approx(0.55 if k in NARROW_VIEW_KEYS else 0.0), k
    assert set(SLOT_PRIOR) == set(LABEL_KEYS)


def test_prior_shifts_attention_towards_the_prior_slots():
    model = build_model(small_cfg(MAIN_SLOTS, head=HEAD_LABEL), pretrained=False).eval()
    images, present = batch(4)
    present[:] = True
    with torch.no_grad():
        model.label_head.slot_prior.zero_()
        flat = model.label_attention(images, present)
        model.label_head.slot_prior.copy_(prior_matrix(MAIN_SLOTS, (), 130.0, 0.55))
        primed = model.label_attention(images, present)
    s, k = 4, model.cfg.k_infer
    mcl, cor = LABEL_KEYS.index("mcl"), MAIN_SLOTS.index("cor_fs")
    on = slice(s + cor * k, s + (cor + 1) * k)
    assert float(primed[:, on, mcl].sum()) > float(flat[:, on, mcl].sum())
    assert torch.allclose(primed[:, :, LABEL_KEYS.index("fracture")],
                          flat[:, :, LABEL_KEYS.index("fracture")], atol=1e-5)


@pytest.mark.parametrize("head", [HEAD_LABEL, HEAD_RESIDUAL, HEAD_XFMR])
def test_per_label_bundle_roundtrip(tmp_path: Path, head):
    cfg = TrainConfig(run_id="t", model=small_cfg(MAIN_SLOTS, head=head))
    model = build_model(cfg.model, pretrained=False).eval()
    save_bundle(model, bundle_meta(cfg, fold=0), tmp_path)
    saved = json.loads((tmp_path / BUNDLE_NAME).read_text())
    assert saved["config"]["model"]["head"] == head
    assert saved["config"]["model"]["slot_prior_strength"] == 0.55
    back, _ = load_bundle(tmp_path, "cpu")
    assert torch.equal(back.label_head.slot_prior, model.label_head.slot_prior)
    images, present = batch(4)
    with torch.no_grad():
        assert torch.equal(model(images, present)[0], back(images, present)[0])


# Forwards of the two depth-1 heads recorded on the server before `per_label_xfmr` existed
# (same pinned_model() setup; the residual gate set to 0.5 so its delta path is exercised).
PINNED_LABEL_LOGITS = [
    0.144675, -0.220496, 1.383727, -1.143056, 1.394978, -1.372787, -2.92793, 1.015957, -1.211938,
    -0.320161, 0.591658, -0.34686, -0.028993, -0.017061, 1.461216, -1.12608, 1.37035, -1.267197,
    -2.803724, 0.935894, -1.067081, -0.409081, 0.620707, -0.190363]
PINNED_RESIDUAL_LOGITS = [
    0.403614, 0.491378, -0.22664, 0.161965, -0.070829, 0.265059, 0.237283, -0.911541, 0.067256,
    0.353262, -0.171523, 0.356057, 0.531402, 0.348253, -0.107824, 0.026419, -0.028836, 0.278565,
    0.143319, -0.793128, -0.016717, 0.290232, -0.274364, 0.375127]


@pytest.mark.parametrize("head,pinned,n_state", [(HEAD_LABEL, PINNED_LABEL_LOGITS, 139),
                                                  (HEAD_RESIDUAL, PINNED_RESIDUAL_LOGITS, 143)])
def test_depth1_heads_are_unchanged_by_the_transformer_head(head, pinned, n_state):
    torch.manual_seed(0)
    model = build_model(small_cfg(MAIN_SLOTS, head=head), pretrained=False).eval()
    if head == HEAD_RESIDUAL:
        with torch.no_grad():
            model.gate.fill_(0.5)
    torch.manual_seed(0)
    images = torch.randint(0, 256, (2, 4, 4, 3, 32, 32), dtype=torch.uint8)
    present = torch.ones(2, 4, dtype=torch.bool)
    present[1, 1] = False
    with torch.no_grad():
        logits, aux = model(images, present)
    assert not hasattr(model, "encoder") and len(model.state_dict()) == n_state
    assert logits.flatten().tolist() == pytest.approx(pinned, abs=2e-5)
    assert float(aux.double().sum()) == pytest.approx(PINNED_SHARED_AUX_SUM, abs=2e-5)


def xfmr_model(**kw):
    cfg = small_cfg(MAIN_SLOTS, head=HEAD_XFMR, **kw)
    return build_model(cfg, pretrained=False)


def test_xfmr_stochastic_depth_is_linear_and_only_active_in_train_mode():
    model = xfmr_model(xfmr_layers=4, xfmr_drop_path=0.15)
    assert model.encoder.drop_path_rates == pytest.approx([0.0, 0.05, 0.10, 0.15])
    assert xfmr_model(xfmr_layers=1).encoder.drop_path_rates == [0.0]
    # Every other dropout off, so any train-mode randomness can only come from drop path.
    quiet = dict(xfmr_attn_drop=0.0, label_attn_drop=0.0, drop_path=0.0)
    images, present = batch(4)
    pf = present.float()
    for dp, differs in ((0.9, True), (0.0, False)):
        torch.manual_seed(0)
        model = xfmr_model(xfmr_layers=3, xfmr_drop_path=dp, **quiet)
        with torch.no_grad():
            filled = model.eval().fill(model.embed(images, present), pf)
            e1, e2 = model.fuse(filled, pf)[0], model.fuse(filled, pf)[0]
            model.encoder.train()
            model.label_head.train()
            torch.manual_seed(1)
            t1 = model.fuse(filled, pf)[0]
            torch.manual_seed(2)
            t2 = model.fuse(filled, pf)[0]
        assert torch.equal(e1, e2)
        assert (not torch.allclose(t1, t2, atol=1e-6)) is differs
        if not differs:
            assert torch.allclose(t1, e1, atol=1e-6)


def test_xfmr_masks_absent_slots_and_padded_triplets():
    torch.manual_seed(0)
    model = xfmr_model().eval()
    images, present = batch(4)
    pf = present.float()
    s, k = 4, 4
    with torch.no_grad():
        filled = model.fill(model.embed(images, present), pf)
        ref, _ = model.fuse(filled, pf)
        # Junk in the triplet tokens of study 1's absent slot 1 must not reach any output.
        junk = filled.clone()
        junk[1, s + 1 * k:s + 2 * k] = torch.randn(k, model.dim) * 3.0
        out, _ = model.fuse(junk, pf)
        assert torch.allclose(out, ref, atol=1e-4)
        # Ragged stacks: slot 0 of study 0 has only 2 real triplets; junk in the 2 padded ones is
        # invisible to the encoder (key padding) and to the read-out (masked logits).
        valid = torch.ones(2, s * k, dtype=torch.bool)
        valid[0, 2:4] = False
        enc = model.encoder(filled, pf, valid)
        base, attn = model.label_head(enc, pf, valid)
        junk = filled.clone()
        junk[0, s + 2:s + 4] = torch.randn(2, model.dim) * 3.0
        enc2 = model.encoder(junk, pf, valid)
        again, _ = model.label_head(enc2, pf, valid)
    assert torch.isfinite(base).all() and torch.allclose(base, again, atol=1e-4)
    assert torch.allclose(attn.sum(dim=1), torch.ones(2, 12), atol=1e-5)
    assert float(attn[0, s + 2:s + 4].max()) < 1e-5
    # Depth positions: the 2 valid triplets of the ragged slot sit at 1/4 and 3/4 of that slot.
    pos = model.encoder.depth_positions(2, k, valid, torch.device("cpu"))
    assert pos[0, :2].tolist() == pytest.approx([0.25, 0.75])
    assert pos[1, :4].tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875])


def test_xfmr_all_missing_study_is_finite():
    model = xfmr_model().eval()
    images, present = batch(4)
    with torch.no_grad():
        logits, aux = model(images, torch.zeros_like(present))
    assert torch.isfinite(logits).all() and torch.isfinite(aux).all()


def test_bundle_rejects_corruption(tmp_path: Path):
    cfg = TrainConfig(run_id="t", model=small_cfg())
    save_bundle(build_model(cfg.model, pretrained=False), bundle_meta(cfg, 0), tmp_path)
    state = load_file(str(tmp_path / WEIGHTS_NAME))
    key = sorted(state)[0]
    state[key] = state[key] + 1.0
    save_file(state, str(tmp_path / WEIGHTS_NAME))
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_bundle(tmp_path, "cpu")
    del state[key]
    save_file(state, str(tmp_path / WEIGHTS_NAME))
    with pytest.raises(RuntimeError, match="missing"):
        load_bundle(tmp_path, "cpu")
