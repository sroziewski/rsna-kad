"""Stage 4 inference (architecture §2.7): cache parity of StudyPreprocessor, robustness
follow-ups F1/F5, ensemble prediction on the CPU, fingerprint aborts, submission validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file, save_file

from conftest import STUDY_A, STUDY_B, UID_ROOT, SyntheticData
from rsna_kad import cache, infer
from rsna_kad.config import CacheConfig, ModelConfig, TrainConfig, params_sha256, to_dict
from rsna_kad.constants import ID_COL, LABEL_KEYS, LABELS, SERIES_COL, SLOTS
from rsna_kad.data import StudyDataset, read_triplets, sample_centres, triplet_indices, view_params
from rsna_kad.infer import (Member, StudyPreprocessor, load_member, predict_studies, prior_vector, safe_read_slice,
                            study_inputs, write_submission)
from rsna_kad.kaggle_export import export_bundle
from rsna_kad.model import build_model, bundle_meta, save_bundle

TINY_SLOTS = ("sag_fs", "cor_fs", "ax_fs")


def tiny_train_cfg(run_id: str = "tiny", slots=TINY_SLOTS) -> TrainConfig:
    cfg = TrainConfig(run_id=run_id)
    cfg.model = ModelConfig(backbone="resnet18", img_px=32, slots=slots, k_train=4, k_infer=4, attn_hidden=16,
                            fusion_hidden=32, grad_checkpoint=False)
    return cfg


def make_tiny_export(root: Path, n_folds: int = 2, with_oof: bool = True) -> Path:
    """runs/tiny/fold{k} bundles (random resnet18 @32 px) -> export dir in the §3.2 layout."""
    run_dirs = []
    for k in range(n_folds):
        torch.manual_seed(k)
        cfg = tiny_train_cfg()
        model = build_model(cfg.model, pretrained=False)
        rd = root / "runs" / "tiny" / f"fold{k}"
        save_bundle(model, bundle_meta(cfg, k, metrics={"oof_weak_auc": 0.5 + k / 100}), rd)
        if with_oof:
            oof = pd.DataFrame({ID_COL: [f"s{i}" for i in range(4)]})
            for j, key in enumerate(LABEL_KEYS):
                oof[f"{key}_pred"] = 0.05 * (j + 1) + 0.01 * k
            oof.to_parquet(rd / "oof.parquet", index=False)
        run_dirs.append(rd)
    out = root / "export" / "tiny"
    export_bundle(run_dirs, out, mode="main", workers=1, user="someone")
    return out


# ------------------------------------------------------------------ preprocessing parity


def _process(synthetic: SyntheticData, tmp: Path, study: str, uid: str, cfg: CacheConfig, plane=None, fs=None):
    spec = synthetic.spec(uid)
    task = {"split": "train", "study": study, "series": uid, "plane": plane or spec.plane,
            "fs": spec.fs if fs is None else fs, "series_dir": str(synthetic.series_dir(study, uid)),
            "npy": str(tmp / study / f"{uid}.npy"), "cfg": to_dict(cfg), "params_sha256": params_sha256(cfg),
            "resume": False}
    return cache.process_series(task)


def test_series_volume_equals_cache_process_series(synthetic: SyntheticData, tmp_path: Path):
    cfg = CacheConfig()
    pre = StudyPreprocessor(cfg, SLOTS)
    for study, specs in synthetic.specs.items():
        for spec in specs:
            rec = _process(synthetic, tmp_path, study, spec.uid, cfg)
            assert rec["ok"], rec
            ref = np.load(tmp_path / study / f"{spec.uid}.npy")
            vol, meta = pre.series_volume(synthetic.series_dir(study, spec.uid))
            assert vol.dtype == np.uint8 and vol.shape == ref.shape
            assert np.array_equal(vol, ref), spec.uid
            assert meta.n_cached == rec["n_cached"] and meta.truncated_slices == rec["truncated_slices"]
            assert meta.p_lo == rec["p_lo"] and meta.p_hi == rec["p_hi"]


def test_study_slots_match_cache_index(synthetic: SyntheticData, tmp_path: Path):
    out = tmp_path / "v1"
    cfg = CacheConfig()
    cache.build_cache(synthetic.data_dir, out, "train", cfg, workers=2)
    index = cache.CacheIndex(out)
    pre = StudyPreprocessor(cfg, SLOTS)
    for study in (STUDY_A, STUDY_B):
        rows = synthetic.series_df()[lambda d: d[ID_COL] == study]
        vols = pre.study(synthetic.data_dir / "train_series" / study, rows)
        expected = {slot: index.series_for("train", study, slot) for slot in SLOTS}
        assert set(vols) == {s for s, c in expected.items() if c}
        for slot, vol in vols.items():
            ref = cache.load_series(out, "train", study, expected[slot][0])
            assert np.array_equal(vol, np.asarray(ref)), slot
    # study A: cor_fs is served by the cor_t1 series (fallback), ax_fs by the 3D series
    a = pre.study(synthetic.data_dir / "train_series" / STUDY_A, synthetic.series_df()[lambda d: d[ID_COL] == STUDY_A])
    assert np.array_equal(a["cor_fs"], a["cor_t1"]) and a["ax_fs"].shape[0] == 40
    # study B: rank 0 of sag_fs is the 12-slice series, not the 8-slice one
    b = pre.study(synthetic.data_dir / "train_series" / STUDY_B, synthetic.series_df()[lambda d: d[ID_COL] == STUDY_B])
    assert b["sag_fs"].shape[0] == 12


def test_truncated_slice_is_filled_not_fatal(synthetic: SyntheticData):
    pre = StudyPreprocessor(CacheConfig(), SLOTS)
    vol, meta = pre.series_volume(synthetic.series_dir(STUDY_A, UID_ROOT + "200001"))
    assert meta.truncated_slices == 1 and vol.shape == (30, 384, 384)


# ------------------------------------------------------------------ F1 / F5 robustness


def test_f5_undecodable_syntax_becomes_missing_slice_then_missing_slot(synthetic: SyntheticData, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch):
    import rsna_kad.dicom as dicom_mod

    real = dicom_mod.pydicom.dcmread

    def broken(path, *a, **kw):
        if kw.get("stop_before_pixels"):
            return real(path, *a, **kw)
        raise RuntimeError("Unable to decompress 'JPEG 2000' as there are no available plugins")

    monkeypatch.setattr(dicom_mod.pydicom, "dcmread", broken)
    path = synthetic.spec(UID_ROOT + "300001").paths[0]
    assert dicom_mod.read_slice(path) is None
    assert safe_read_slice(path) is None
    rec = _process(synthetic, tmp_path, STUDY_B, UID_ROOT + "300001", CacheConfig())
    assert not rec["ok"] and "ValueError" in rec["error"] and rec["slot"] == "sag_fs"
    pre = StudyPreprocessor(CacheConfig(), ("sag_fs", "ax_fs"))
    rows = synthetic.series_df()[lambda d: d[ID_COL] == STUDY_B]
    assert pre.study(synthetic.data_dir / "train_series" / STUDY_B, rows) == {}


def test_f5_notimplemented_is_caught_too(synthetic: SyntheticData, monkeypatch: pytest.MonkeyPatch):
    import rsna_kad.dicom as dicom_mod

    monkeypatch.setattr(dicom_mod.pydicom, "dcmread", lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError("x")))
    assert dicom_mod.read_slice(synthetic.spec(UID_ROOT + "300001").paths[0]) is None


def test_f1_unknown_plane_is_an_error_row_not_a_crash(synthetic: SyntheticData, tmp_path: Path):
    cfg = CacheConfig()
    bad = _process(synthetic, tmp_path, STUDY_B, UID_ROOT + "300001", cfg, plane="Oblique")
    assert not bad["ok"] and "ValueError" in bad["error"] and bad["slot"] == ""
    good = _process(synthetic, tmp_path, STUDY_B, UID_ROOT + "300003", cfg)
    df, _rules = cache._finalize(pd.DataFrame([bad, good]), None, cfg)
    assert len(df) == 2 and df.loc[df[SERIES_COL] == UID_ROOT + "300001", "rank"].item() == -1
    assert df.loc[df[SERIES_COL] == UID_ROOT + "300003", "selected"].item()
    # the preprocessor skips the row and still serves the other slots
    rows = synthetic.series_df()[lambda d: d[ID_COL] == STUDY_B].copy()
    rows.loc[rows[SERIES_COL] == UID_ROOT + "300001", "Anatomical_Plane"] = "Oblique"
    vols = StudyPreprocessor(cfg, SLOTS).study(synthetic.data_dir / "train_series" / STUDY_B, rows)
    assert vols["sag_fs"].shape[0] == 8 and "ax_fs" in vols


def test_missing_series_dir_and_empty_study(synthetic: SyntheticData, tmp_path: Path):
    rows = synthetic.series_df()[lambda d: d[ID_COL] == STUDY_B]
    pre = StudyPreprocessor(CacheConfig(), SLOTS)
    assert pre.study(tmp_path / "nowhere", rows) == {}
    assert pre.study(tmp_path / "nowhere", rows.head(0)) == {}


# ------------------------------------------------------------------ inputs and members


def test_study_inputs_shapes_and_views():
    vols = {"sag_fs": np.random.default_rng(0).integers(0, 255, (9, 64, 64), dtype=np.uint8)}
    images, present = study_inputs(vols, (32, TINY_SLOTS, 4, (0.0, 0.5), (130.0,) * 3))
    assert images.shape == (2, 3, 4, 3, 32, 32) and images.dtype == np.uint8
    assert present.tolist() == [True, False, False]
    assert images[0, 1].sum() == 0 and images[0, 0].sum() > 0
    assert not np.array_equal(images[0, 0], images[1, 0])  # the z-offset view samples other slices


def test_study_inputs_crops_a_repeated_slot():
    vols = {"sag_fs": np.random.default_rng(0).integers(0, 255, (9, 64, 64), dtype=np.uint8)}
    slots = ("sag_fs", "cor_fs", "sag_fs")
    images, present = study_inputs(vols, (32, slots, 4, (0.0,), (130.0, 130.0, 90.0)), 130.0)
    assert present.tolist() == [True, False, True]
    assert not np.array_equal(images[0, 0], images[0, 2])  # the 90 mm view is a different picture
    ref = read_triplets(np.asarray(vols["sag_fs"]), triplet_indices(sample_centres(9, 4, None), 9), 32,
                        view_params(130.0, 90.0))
    assert np.array_equal(images[0, 2], ref)
    # A whole-field view must stay on the untouched path: same bytes as before view crops existed.
    plain, _ = study_inputs(vols, (32, TINY_SLOTS, 4, (0.0,), (130.0,) * 3))
    assert np.array_equal(images[0, 0], plain[0, 0])


def test_load_member_rejects_tampering(tmp_path: Path):
    out = make_tiny_export(tmp_path, n_folds=1, with_oof=False)
    doc = json.loads((out / "ensemble.json").read_text())
    prefix = out / doc["members"][0]["path"]
    model, meta = load_member(prefix, "cpu")
    assert meta["fold"] == 0 and model.training is False
    weights = Path(str(prefix) + ".safetensors")
    state = load_file(str(weights))
    key = sorted(k for k, v in state.items() if v.is_floating_point())[0]
    dropped = {k: v for k, v in state.items() if k != key}
    save_file(dropped, str(weights))
    with pytest.raises(RuntimeError, match="missing"):
        load_member(prefix, "cpu")
    state[key] = state[key] + 1.0
    save_file(state, str(weights))
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_member(prefix, "cpu")


def test_view_crop_member_inputs_equal_training_inputs(synthetic: SyntheticData, tmp_path: Path):
    """The stage-4 parity contract (§2.7) extended to view crops: what the notebook feeds a
    view-crop member must be what StudyDataset fed it during training, byte for byte."""
    ccfg = CacheConfig()
    cache_dir = tmp_path / "v1"
    cache.build_cache(synthetic.data_dir, cache_dir, "train", ccfg, workers=2)
    cfg = tiny_train_cfg(slots=("sag_fs", "cor_fs", "ax_fs", "sag_fs"))
    cfg.model.img_px = ccfg.px
    cfg.model.view_crop_mm = (130.0, 130.0, 130.0, 90.0)
    meta = bundle_meta(cfg, 0)
    member = Member("unused", tta=False, meta=meta)
    spec = member.spec
    assert spec[1] == ("sag_fs", "cor_fs", "ax_fs", "sag_fs") and spec[4] == (130.0, 130.0, 130.0, 90.0)

    pre = StudyPreprocessor(ccfg, tuple(dict.fromkeys(spec[1])))
    for study in (STUDY_A, STUDY_B):
        rows = synthetic.series_df()[lambda d: d[ID_COL] == study]
        vols = pre.study(synthetic.data_dir / "train_series" / study, rows)
        images, present = study_inputs(vols, spec, ccfg.crop_mm)
        item = StudyDataset(cache_dir, [study], None, cfg, train=False)[0]
        assert present.tolist() == item["present"].tolist()
        assert np.array_equal(images[0], item["images"].numpy()), study
        assert not np.array_equal(images[0, 0], images[0, 3])  # 130 mm and 90 mm really differ


def test_load_member_refuses_a_view_bundle_without_view_inputs(tmp_path: Path):
    cfg = tiny_train_cfg(slots=("sag_fs", "cor_fs", "sag_fs"))
    cfg.model.view_crop_mm = (130.0, 130.0, 90.0)
    rd = tmp_path / "runs" / "views" / "fold0"
    save_bundle(build_model(cfg.model, pretrained=False), bundle_meta(cfg, 0), rd)
    out = tmp_path / "export" / "views"
    export_bundle([rd], out, mode="main", workers=1, user="someone")  # verify=True path
    prefix = out / json.loads((out / "ensemble.json").read_text())["members"][0]["path"]
    with pytest.raises(RuntimeError, match="view crops"):
        load_member(prefix, "cpu")
    model, meta = load_member(prefix, "cpu", views=True)
    assert model.slots == ("sag_fs", "cor_fs", "sag_fs")
    # A bundle that lost its crops keeps a repeated slot that nothing can tell apart: refuse it.
    meta["config"]["model"]["view_crop_mm"] = []
    with pytest.raises(ValueError, match="repeat"):
        load_member(prefix, "cpu", meta, views=True)


def test_prior_vector():
    assert prior_vector(None).tolist() == [0.5] * 12
    assert prior_vector(0.2).tolist() == [0.2] * 12
    with pytest.raises(ValueError):
        prior_vector([0.1, 0.2])


# ------------------------------------------------------------------ end to end (CPU)


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_tiny_export(tmp_path_factory.mktemp("tiny"))


def test_predict_studies_cpu_and_submission(synthetic: SyntheticData, tiny_export: Path, tmp_path: Path):
    doc = json.loads((tiny_export / "ensemble.json").read_text())
    members = infer.members_from_ensemble(tiny_export, doc)
    assert len(members) == 2 and all(m.tta for m in members)
    series_df = synthetic.series_df("test")
    study_dirs = {STUDY_B: str(synthetic.data_dir / "test_series" / STUDY_B), "ghost": str(tmp_path / "ghost")}
    pred = predict_studies(members, study_dirs, series_df, devices=["cpu"], workers=1, prior=doc["prior"])
    assert list(pred[ID_COL]) == [STUDY_B, "ghost"]  # ordered by file count, the empty one last
    row = pred.set_index(ID_COL).loc[STUDY_B]
    assert row["error"] == "" and row["n_slots"] == 3
    probs = row[list(LABELS)].to_numpy(dtype=float)
    assert probs.min() >= 0 and probs.max() <= 1 and np.isfinite(probs).all()
    ghost = pred.set_index(ID_COL).loc["ghost"]
    assert ghost["error"] and np.allclose(ghost[list(LABELS)].to_numpy(dtype=float), doc["prior"])
    # the ensemble is the plain mean of the members' (TTA-averaged) sigmoids
    single = [predict_studies([m], {STUDY_B: study_dirs[STUDY_B]}, series_df, devices=["cpu"], workers=1)
              for m in members]
    mean = np.mean([s.set_index(ID_COL).loc[STUDY_B, list(LABELS)].to_numpy(dtype=float) for s in single], axis=0)
    assert np.allclose(mean, probs, atol=1e-5)

    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({ID_COL: ["ghost", STUDY_B], **{lab: [0.5, 0.5] for lab in LABELS}}).to_csv(sample, index=False)
    out = write_submission(pred, sample, tmp_path / "sub" / "submission.csv")
    got = pd.read_csv(tmp_path / "sub" / "submission.csv", dtype={ID_COL: str})
    assert list(got.columns) == [ID_COL, *LABELS] and list(got[ID_COL]) == ["ghost", STUDY_B]
    assert np.allclose(got[list(LABELS)].to_numpy()[1], probs, atol=1e-5) and len(out) == 2


def test_predict_studies_fp32_matches_and_no_tta(synthetic: SyntheticData, tiny_export: Path):
    doc = json.loads((tiny_export / "ensemble.json").read_text())
    series_df = synthetic.series_df("test")
    dirs = {STUDY_B: str(synthetic.data_dir / "test_series" / STUDY_B)}
    a = predict_studies(infer.members_from_ensemble(tiny_export, doc, tta=False), dirs, series_df, devices=["cpu"],
                        workers=1, fp16=False)
    b = predict_studies(infer.members_from_ensemble(tiny_export, doc, tta=False), dirs, series_df, devices=["cpu"],
                        workers=1, fp16=True)
    assert np.allclose(a[list(LABELS)].to_numpy(), b[list(LABELS)].to_numpy(), atol=1e-6)


def test_wall_clock_guard_returns_every_study(synthetic: SyntheticData, tiny_export: Path):
    members = infer.members_from_ensemble(tiny_export)
    dirs = {STUDY_B: str(synthetic.data_dir / "test_series" / STUDY_B), "later": str(synthetic.data_dir / "none")}
    pred = predict_studies(members, dirs, synthetic.series_df("test"), devices=["cpu"], workers=1, deadline_s=-1.0)
    assert len(pred) == 2 and set(pred["error"]) <= {"", "unfinished at deadline", "no usable slot"}
    assert pred[list(LABELS)].notna().all().all()


def test_write_submission_validation(tmp_path: Path):
    sample = tmp_path / "sample_submission.csv"
    ids = ["a", "b", "c"]
    pd.DataFrame({ID_COL: ids, **{lab: [0.5] * 3 for lab in LABELS}}).to_csv(sample, index=False)
    pred = pd.DataFrame({ID_COL: ["c", "a"], **{lab: [0.1, 0.9] for lab in LABELS}})
    with pytest.raises(ValueError, match="no prediction"):
        write_submission(pred, sample, tmp_path / "s.csv")
    out = write_submission(pred, sample, tmp_path / "s.csv", prior=0.3)
    assert list(out[ID_COL]) == ids and out.loc[1, "ACL"] == pytest.approx(0.3)
    bad = pred.copy()
    bad.loc[0, "MCL"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        write_submission(bad, sample, tmp_path / "s.csv", prior=0.3)
    bad.loc[0, "MCL"] = 1.5
    with pytest.raises(ValueError, match="outside"):
        write_submission(bad, sample, tmp_path / "s.csv", prior=0.3)
    with pytest.raises(ValueError, match="unexpected columns"):
        pd.DataFrame({ID_COL: ids, "ACL": [0.5] * 3}).to_csv(sample, index=False)
        write_submission(pred, sample, tmp_path / "s.csv", prior=0.3)


def test_find_dir_discovers_kaggle_mounts(tmp_path: Path):
    root = tmp_path / "input"
    comp = root / "competitions" / "rsna-knee-abnormality-detection"
    (comp / "test_series" / "study" / "series").mkdir(parents=True)
    (comp / "sample_submission.csv").write_text("x")
    ds = root / "datasets" / "user" / "rsna-kad-weights"
    (ds / "wheels").mkdir(parents=True)
    (ds / "ensemble.json").write_text("{}")
    assert infer.find_dir(root / "rsna-kad-weights", "ensemble.json", root=str(root)) == ds
    assert infer.find_dir(root / "nope", "sample_submission.csv", root=str(root)) == comp
    assert infer.find_dir(ds, "ensemble.json", root=str(root)) == ds  # preferred wins when it has the marker
    assert infer.find_dir(root / "nope", "missing.txt", root=str(root)) == root / "nope"
    listing = infer.describe_inputs(str(root))
    assert "datasets/" in listing and "competitions/rsna-knee-abnormality-detection/" in listing
    assert not any("study" in item for item in listing)  # study trees are not walked


def test_study_record_keeps_metadata_only():
    """Regression for the Kaggle OOM of kernel v2: the driver must not retain the triplets."""
    big = {(32, TINY_SLOTS, 4, (0.0,)): (np.zeros((1, 3, 4, 3, 32, 32), np.uint8), np.ones(3, bool))}
    meta, inputs = infer.study_record({"study": "s", "inputs": big, "n_slots": 3, "error": "", "seconds": 0.1})
    assert inputs is big and "inputs" not in meta and meta["has_inputs"] is True and meta["n_slots"] == 3
    meta2, inputs2 = infer.study_record({"study": "t", "inputs": {}, "n_slots": 0, "error": "", "seconds": 0.0})
    assert inputs2 == {} and meta2["has_inputs"] is False


def test_members_split_across_devices():
    ms = [Member(f"m{i}") for i in range(5)]
    infer.assign_devices(ms, ["cuda:0", "cuda:1"])
    assert [m.device for m in ms] == ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]
    assert infer.available_devices(["cpu"]) == ["cpu"]
