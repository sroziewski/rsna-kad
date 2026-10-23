"""Full-data retrains (train.py --fold full): all non-gold studies trained, gold-58 as the only
held-out rows, bundle fold "full", and the export guard / export / inference accepting such a
run as one complete member."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import STUDY_B, SyntheticData
from rsna_kad import infer, runguard
from rsna_kad import train as tr
from rsna_kad.constants import ID_COL, LABELS
from rsna_kad.data import synthetic_targets
from rsna_kad.kaggle_export import export_bundle, member_name
from rsna_kad.model import build_model, bundle_meta, load_bundle, save_bundle
from test_infer import tiny_train_cfg
from test_runguard import _load_script, make_fold
from test_train import env, patch_dirs  # noqa: F401  (fixtures)

# ------------------------------------------------------------------ training


def test_paths_and_split(env):  # noqa: F811
    cfg = env["cfg"]
    assert tr.targets_path(cfg, "full").name == "full.parquet"
    assert tr.targets_path(cfg, 3).name == "fold3.parquet"
    assert tr.run_dir_for(cfg, "full", "runs") == Path("runs") / cfg.run_id / "full"
    assert tr.run_dir_for(cfg, 2, "runs") == Path("runs") / cfg.run_id / "fold2"
    folds = pd.read_csv(env["root"] / "folds.csv")
    folds["is_gold"] = folds["is_gold"].astype(bool)
    all_cached = {"S1", "S2", "S3", "S4", "S5"}
    train_ids, val_ids = tr.split_full(folds, env["targets"], all_cached)
    assert train_ids == ["S1", "S2", "S3", "S4"] and val_ids == ["S5"]
    # gold stays out even when it has a targets row
    with_gold_row = pd.concat([env["targets"], synthetic_targets(["S5"], seed=3)])
    assert "S5" not in tr.split_full(folds, with_gold_row, all_cached)[0]


def test_train_fold_full_end_to_end(env, tmp_path: Path):  # noqa: F811
    cfg = env["cfg"]
    cfg_full = copy.deepcopy(cfg)
    cfg_full.select_epoch = "last"
    env["targets"].to_parquet(Path(cfg.targets_dir) / cfg.targets_version / "full.parquet", index=False)
    run_dir = tr.run_dir_for(cfg_full, "full", tmp_path / "runs")
    metrics = tr.train_fold(cfg_full, "full", run_dir, device="cpu", pretrained=False)
    assert not (run_dir / tr.LAST_NAME).exists()
    assert metrics["n_train"] == 4 and metrics["n_val"] == 1 and metrics["epoch"] == cfg.epochs - 1
    # no weak/strict metric exists for a full-data run (write_json stores NaN as null)
    assert all(metrics[k] is None or math.isnan(metrics[k]) for k in ("oof_weak_auc", "oof_strict_auc"))
    assert metrics["n_gold"] == 1 and "gold_auc" in metrics
    oof = pd.read_parquet(run_dir / "oof.parquet")
    assert list(oof[ID_COL]) == ["S5"] and (oof["fold"] == "full").all()
    _, meta = load_bundle(run_dir, "cpu")
    assert meta["fold"] == "full" and meta["metrics"]["epoch"] == cfg.epochs - 1
    # the export guard's completeness check accepts it as a finished member
    assert runguard.fold_problems(run_dir) == []
    assert runguard.export_problems([run_dir.parent], range(5), procs=[]) == []


def test_full_run_refuses_best_epoch_selection(env, tmp_path: Path):  # noqa: F811
    cfg = copy.deepcopy(env["cfg"])
    cfg.select_epoch = "best"
    with pytest.raises(ValueError, match="select_epoch: last"):
        tr.train_fold(cfg, "full", tmp_path / "x" / "full", device="cpu", pretrained=False)


def test_train_script_fold_and_seed_arguments():
    mod = _load_script("train")
    assert mod._fold_arg("full") == "full" and mod._fold_arg("3") == 3
    with pytest.raises(ValueError):
        mod._fold_arg("four")
    assert runguard.seeded_run_id("arm_full", 2126) == "arm_full_s2126"


# ------------------------------------------------------------------ export guard


def make_full(run: Path, **kw) -> Path:
    """A finished full-data member at <run>/full (make_fold's layout, fold 'full')."""
    fd = make_fold(run, 0, **kw)
    target = run / "full"
    fd.rename(target)
    meta = json.loads((target / "bundle.json").read_text())
    meta["fold"] = "full"
    (target / "bundle.json").write_text(json.dumps(meta))
    return target


def test_full_run_is_a_complete_member_set(tmp_path: Path):
    run = tmp_path / "runs" / "arm_full_s2026"
    make_full(run)
    assert runguard.is_full_run(run)
    assert runguard.member_dirs(run, range(5)) == [run / "full"]
    assert runguard.export_problems([run], range(5), procs=[]) == []
    # a fold run next to it keeps every existing rule: fold3 missing is still an error
    fold_run = tmp_path / "runs" / "arm"
    for k in (0, 1, 2, 4):
        make_fold(fold_run, k)
    errors = runguard.export_problems([run, fold_run], range(5), procs=[])
    assert len(errors) == 1 and "fold3" in errors[0]


@pytest.mark.parametrize("case", ["last_pt", "early", "stray_fold", "smoke", "limit"])
def test_full_run_keeps_every_refusal(tmp_path: Path, case: str):
    name = "arm_smoke_full" if case == "smoke" else "arm_full"
    run = tmp_path / "runs" / name
    make_full(run, last_pt=case == "last_pt", epoch=3 if case == "early" else None,
              limit=200 if case == "limit" else None)
    if case == "stray_fold":
        make_fold(run, 0)
    errors = runguard.export_problems([run], range(5), procs=[])
    assert errors, case
    if case in ("smoke", "limit"):
        assert all("smoke/sweep" in e for e in errors)
        assert runguard.export_problems([run], range(5), procs=[], allow_smoke=True) == []


# ------------------------------------------------------------------ export + inference


def test_full_member_exports_and_predicts(synthetic: SyntheticData, tmp_path: Path):
    runs = []
    for fold in ("full", 0):
        cfg = tiny_train_cfg("tinyfull" if fold == "full" else "tiny")
        meta = bundle_meta(cfg, 0, metrics={"oof_weak_auc": 0.5})
        meta["fold"] = fold
        rd = tmp_path / "runs" / cfg.run_id / ("full" if fold == "full" else "fold0")
        save_bundle(build_model(cfg.model, pretrained=False), meta, rd)
        runs.append(rd)
    assert member_name(json.loads((runs[0] / "bundle.json").read_text())) == "tinyfull_full"
    assert member_name(json.loads((runs[1] / "bundle.json").read_text())) == "tiny_f0"
    out = tmp_path / "export" / "x"
    doc = export_bundle(runs, out, mode="efficiency", workers=1, user="someone")
    assert [m["path"] for m in doc["members"]] == ["models/tinyfull_full", "models/tiny_f0"]
    # infer.py needs no change: a member is a path; the fold is never read
    members = infer.members_from_ensemble(out, json.loads((out / "ensemble.json").read_text()))
    series_df = synthetic.series_df("test")
    dirs = {STUDY_B: str(synthetic.data_dir / "test_series" / STUDY_B)}
    pred = infer.predict_studies(members, dirs, series_df, devices=["cpu"], workers=1)
    probs = pred.set_index(ID_COL).loc[STUDY_B, list(LABELS)].to_numpy(dtype=float)
    assert pred.set_index(ID_COL).loc[STUDY_B, "error"] == "" and np.isfinite(probs).all()


def test_export_script_ships_a_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from rsna_kad import kaggle_export as ke
    cfg = tiny_train_cfg("tinyfull")
    cfg.select_epoch = "last"
    meta = bundle_meta(cfg, 0, metrics={"epoch": cfg.epochs - 1})
    meta["fold"] = "full"
    rd = tmp_path / "runs" / "tinyfull" / "full"
    save_bundle(build_model(cfg.model, pretrained=False), meta, rd)
    (rd / "metrics.json").write_text(json.dumps({"epoch": cfg.epochs - 1, "last_epoch": {"epoch": cfg.epochs - 1}}))
    monkeypatch.setattr(ke, "build_notebook", lambda p: Path(p))
    monkeypatch.setattr(ke, "write_kernel_metadata", lambda d, u: Path(d))
    rc = _load_script("export_kaggle").main(["--run", str(rd.parent), "--name", "f", "--user", "u",
                                              "--export-root", str(tmp_path / "export"), "--no-verify"])
    assert rc == 0
    doc = json.loads((tmp_path / "export" / "f" / "ensemble.json").read_text())
    assert [m["path"] for m in doc["members"]] == ["models/tinyfull_full"]
