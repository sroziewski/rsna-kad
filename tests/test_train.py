"""train_fold end-to-end on CPU over the tiny cache: outputs, resume, OOF assembly, metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_kad import train as tr
from rsna_kad.constants import ID_COL, LABEL_KEYS, LABELS
from rsna_kad.cv import FOLD_COLUMNS
from rsna_kad.data import StudyDataset, synthetic_targets
from rsna_kad.model import load_bundle
from test_data import make_tiny_cache, tiny_cfg

# S1, S2, S4 train (fold 1); S3 weak validation and S5 gold (fold 0).
FOLDS = pd.DataFrame({ID_COL: ["S1", "S2", "S3", "S4", "S5"], "fold": [1, 1, 0, 1, 0],
                      "site_key": ["a", "b", "c", "a", "d"], "lang": ["en"] * 5,
                      "is_gold": [False, False, False, False, True]})[FOLD_COLUMNS]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("train")
    cache_dir = make_tiny_cache(root / "cache")
    folds_csv = root / "folds.csv"
    FOLDS.to_csv(folds_csv, index=False)
    targets = synthetic_targets(["S1", "S2", "S3", "S4"], seed=2)
    for k in (0, 1):
        (root / "targets" / "t").mkdir(parents=True, exist_ok=True)
        targets.to_parquet(root / "targets" / "t" / f"fold{k}.parquet", index=False)
    labels = pd.DataFrame({ID_COL: ["S1", "S2", "S3", "S4", "S5"]})
    for k in LABEL_KEYS:
        labels[f"{k}_state"] = ["affirmed", "negated", "affirmed", "absent", "negated"]
    labels.to_parquet(root / "labels.parquet", index=False)
    data_dir = root / "data"
    data_dir.mkdir()
    gold = pd.DataFrame({ID_COL: ["S5", "S1"], **{lab: [1.0, np.nan] for lab in LABELS}})
    gold.to_csv(data_dir / "train.csv", index=False)
    cfg = tiny_cfg(folds_csv=str(folds_csv), targets_dir=str(root / "targets"), targets_version="t",
                   labels_parquet=str(root / "labels.parquet"), num_workers=0, epochs=2, batch_studies=2,
                   mixup_p=1.0, eval_batch_studies=2)
    return {"root": root, "cache": cache_dir.parent, "cfg": cfg, "targets": targets, "labels": labels}


@pytest.fixture(autouse=True)
def patch_dirs(env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tr, "CACHE_DIR", str(env["cache"]))
    monkeypatch.setattr(tr, "DATA_DIR", str(env["root"] / "data"))


def test_split_studies_excludes_gold_and_limits(env):
    folds = pd.read_csv(env["root"] / "folds.csv")
    folds["is_gold"] = folds["is_gold"].astype(bool)
    train_ids, val_ids = tr.split_studies(folds, 0, env["targets"], {"S1", "S2", "S3", "S4", "S5"})
    assert train_ids == ["S1", "S2", "S4"] and val_ids == ["S3", "S5"]
    tr_lim, val_lim = tr.split_studies(folds, 0, env["targets"], {"S1", "S2", "S3", "S4", "S5"}, limit=2)
    assert len(tr_lim) == 2 and "S5" in val_lim and len(val_lim) == 2
    tr2, val2 = tr.split_studies(folds, 1, env["targets"], {"S1", "S3"})  # cache membership filters
    assert tr2 == ["S3"] and val2 == ["S1"]


def test_schedule_and_ema():
    assert tr.cosine_warmup(0, 100, 10) == pytest.approx(0.1)
    assert tr.cosine_warmup(9, 100, 10) == pytest.approx(1.0)
    assert tr.cosine_warmup(55, 100, 10) == pytest.approx(0.5)
    assert tr.cosine_warmup(100, 100, 10) == pytest.approx(0.0)
    lin = torch.nn.Linear(2, 2)
    ema = tr.Ema(lin, 0.999)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    before = ema.module.weight.clone()
    ema.update()
    d = 2.0 / 11.0
    assert torch.allclose(ema.module.weight, d * before + (1 - d) * 1.0)
    state = ema.state_dict()
    ema2 = tr.Ema(lin, 0.999)
    ema2.load_state_dict(state)
    assert ema2.step == 1 and torch.equal(ema2.module.weight, ema.module.weight)


def test_evaluate_metrics(env):
    pred = pd.DataFrame({ID_COL: ["S1", "S2", "S3", "S4", "S5"]})
    for k in LABEL_KEYS:
        pred[f"{k}_pred"] = [0.9, 0.1, 0.8, 0.2, 0.5]
    strict, per = tr.strict_auc(pred, env["labels"])
    assert strict == 1.0 and set(per) == set(LABEL_KEYS)
    targets = env["targets"].copy()
    for k in LABEL_KEYS:
        targets[f"{k}_target"] = [0.92, 0.04, 0.92, 0.04]
        targets[f"{k}_weight"] = 1.0
    m = tr.evaluate(pred, targets, tr.gold_frame_or_none(), env["labels"], n_boot=5)
    assert m["oof_weak_auc"] == 1.0 and m["oof_strict_auc"] == 1.0 and m["n_weak"] == 4
    assert m["n_gold"] == 1 and np.isnan(m["gold_auc"])  # one gold row: AUC undefined, reported as NaN
    assert "gold_ci95" in m and "per_label" in m


def test_train_fold_end_to_end(env, tmp_path: Path):
    cfg = env["cfg"]
    run_dir = tmp_path / "runs" / cfg.run_id / "fold0"
    metrics = tr.train_fold(cfg, 0, run_dir, device="cpu", pretrained=False)
    for name in ("model.safetensors", "bundle.json", "oof.parquet", "metrics.json", "config.yaml", "train.log"):
        assert (run_dir / name).exists(), name
    assert not (run_dir / tr.LAST_NAME).exists()
    assert len(metrics["history"]) == 2 and metrics["epoch"] in (0, 1)
    assert metrics["n_train"] == 3 and metrics["n_val"] == 2
    for key in ("oof_weak_auc", "oof_strict_auc", "gold_auc", "gold_ci95", "per_label"):
        assert key in metrics
    hist = metrics["history"][0]
    assert hist["img_per_s"] > 0 and hist["train_s"] > 0 and np.isfinite(hist["train_loss"])
    oof = pd.read_parquet(run_dir / "oof.parquet")
    assert sorted(oof[ID_COL]) == ["S3", "S5"] and (oof["fold"] == 0).all()
    preds = oof[tr.PRED_COLS].to_numpy()
    assert preds.shape == (2, 12) and (0 <= preds).all() and (preds <= 1).all()
    model, meta = load_bundle(run_dir, "cpu")
    assert meta["fold"] == 0 and meta["run_id"] == cfg.run_id and meta["metrics"]["epoch"] == metrics["epoch"]
    assert meta["config"]["model"]["backbone"] == "resnet18"
    # the saved bundle reproduces the stored OOF predictions
    ds = StudyDataset(env["cache"] / "v1", ["S3", "S5"], None, cfg, train=False)
    again = tr.predict_studies_cached(model, ds, batch_size=2, num_workers=0, device="cpu", dtype=torch.float32)
    merged = oof.merge(again, on=ID_COL, suffixes=("", "_again"))
    for c in tr.PRED_COLS:
        assert np.allclose(merged[c], merged[c + "_again"], atol=1e-5)
    tta = tr.predict_studies_cached(model, ds, tta=True, batch_size=2, num_workers=0, device="cpu",
                                    dtype=torch.float32)
    assert len(tta) == 2 and tta[tr.PRED_COLS].to_numpy().max() <= 1.0
    m = tr.assemble_oof(tmp_path / "runs", cfg, folds=[0])
    root = tmp_path / "runs" / cfg.run_id
    assert (root / "oof.parquet").exists() and (root / "metrics.json").exists() and (root / "config.yaml").exists()
    assert m["folds"] == [0] and m["n_rows"] == 2 and m["n_weak"] == 1
    assert json.loads((root / "metrics.json").read_text())["folds"] == [0]


def test_select_epoch_policy(env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import dataclasses
    curve = [0.6, 0.9, 0.7]  # weak AUC per epoch: best = epoch 1, last = epoch 2
    original = tr.evaluate

    def scripted_evaluate(pred, targets, gold, labels, n_boot=1000):
        m = original(pred, targets, gold, labels, n_boot=n_boot)
        m["oof_weak_auc"] = curve[len(calls)]
        calls.append(m)
        return m

    monkeypatch.setattr(tr, "evaluate", scripted_evaluate)
    for policy in ("last", "best"):
        calls: list = []
        cfg = dataclasses.replace(env["cfg"], select_epoch=policy, epochs=3)
        run_dir = tmp_path / policy / "fold0"
        metrics = tr.train_fold(cfg, 0, run_dir, device="cpu", pretrained=False)
        hist = metrics["history"]
        assert [h["oof_weak_auc"] for h in hist] == curve and metrics["select_epoch"] == policy
        assert metrics["last_epoch"]["epoch"] == 2 and metrics["best_epoch"]["epoch"] == 1
        assert metrics["last_epoch"]["oof_weak_auc"] == 0.7 and metrics["best_epoch"]["oof_weak_auc"] == 0.9
        assert (metrics["epoch"], metrics["oof_weak_auc"]) == ((2, 0.7) if policy == "last" else (1, 0.9))
        _, meta = load_bundle(run_dir, "cpu")
        assert meta["metrics"]["epoch"] == metrics["epoch"] and meta["metrics"]["select_epoch"] == policy
    with pytest.raises(ValueError, match="select_epoch"):
        tr.train_fold(dataclasses.replace(env["cfg"], select_epoch="median"), 0, tmp_path / "bad", device="cpu",
                      pretrained=False)


def test_train_fold_resumes(env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = env["cfg"]
    run_dir = tmp_path / "runs" / cfg.run_id / "fold0"
    original = tr.Ema.update

    def crash_in_second_epoch(self):
        original(self)
        if self.step == 2:
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(tr.Ema, "update", crash_in_second_epoch)
    with pytest.raises(RuntimeError, match="simulated"):
        tr.train_fold(cfg, 0, run_dir, device="cpu", pretrained=False)
    assert (run_dir / tr.LAST_NAME).exists()
    first = json.loads((run_dir / "metrics.json").read_text())
    assert len(first["history"]) == 1
    monkeypatch.setattr(tr.Ema, "update", original)
    metrics = tr.train_fold(cfg, 0, run_dir, device="cpu", pretrained=False)
    assert len(metrics["history"]) == 2 and metrics["history"][0] == first["history"][0]
    assert not (run_dir / tr.LAST_NAME).exists()
    assert "resumed" in (run_dir / "train.log").read_text()
    fresh = tr.train_fold(cfg, 0, tmp_path / "runs2", device="cpu", pretrained=False, resume=False)
    assert len(fresh["history"]) == 2
