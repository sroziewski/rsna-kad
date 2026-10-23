#!/usr/bin/env python3
"""Stage-4 acceptance measurements (architecture §4 row 4, §2.4 inference precision).

  python scripts/bench_infer.py parity --n 20                       # StudyPreprocessor vs cache v1 (uint8 equality)
  python scripts/bench_infer.py precision --fold 0 --n 200 --gpu 1  # fp16 vs fp32 macro AUC on cached val studies
  python scripts/bench_infer.py kinfer --config configs/eff_effv2s_256.yaml --k 12 24 --gpu 1
  python scripts/bench_infer.py time --weights export/mvp_v1 --device cuda:1 [--device cuda:0] [--repeat 2]

`time` runs the real raw-DICOM path on the public test studies and projects 1,300 studies on
Kaggle's T4s (assumed 4.5x slower than an A6000, §2.7). Results are printed as JSON and, with
--out, written to a file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

T4_FACTOR = 4.5
N_TEST_STUDIES = 1300


def parity(args) -> dict:
    import numpy as np
    import pandas as pd
    from rsna_kad.cache import read_index, read_manifest, series_path
    from rsna_kad.config import CacheConfig, from_dict
    from rsna_kad.constants import ID_COL, SERIES_COL, SLOTS
    from rsna_kad.infer import StudyPreprocessor

    cache_dir = Path(args.cache_dir)
    cfg = from_dict(CacheConfig, read_manifest(cache_dir)["params"])
    idx = read_index(cache_dir)
    idx = idx[(idx["split"] == "train") & idx["ok"]]
    rng = np.random.default_rng(args.seed)
    studies = sorted(idx[ID_COL].unique())
    picks = sorted(rng.choice(studies, size=min(args.n, len(studies)), replace=False))
    series_df = pd.read_csv(Path(args.data_dir) / "train_series.csv", dtype=str)
    pre = StudyPreprocessor(cfg, slots=tuple(args.slots) if args.slots else SLOTS)
    per_series, n_equal = [], 0
    n_vox = 0
    t0 = time.monotonic()
    for study in picks:
        rows = series_df[series_df[ID_COL] == study]
        cached_rows = idx[idx[ID_COL] == study]
        for r in cached_rows.itertuples(index=False):
            uid = getattr(r, SERIES_COL)
            ref = np.load(series_path(cache_dir, "train", study, uid), mmap_mode="r")
            vol, _meta = pre.series_volume(Path(args.data_dir) / "train_series" / study / uid)
            same_shape = vol.shape == ref.shape
            frac = float((vol == np.asarray(ref)).mean()) if same_shape else 0.0
            per_series.append({ID_COL: study, SERIES_COL: uid, "shape_ok": same_shape, "equal_frac": frac,
                               "max_abs_diff": int(np.abs(vol.astype(np.int16) - np.asarray(ref).astype(np.int16)).max()) if same_shape else -1})
            n_equal += int(same_shape and frac == 1.0)
            n_vox += int(np.asarray(ref).size)
        # slot selection parity: the chosen series per slot must match the cache's rank-0 series
        vols = pre.study(Path(args.data_dir) / "train_series" / study, rows, study)
        for slot, vol in vols.items():
            sel = cached_rows[(cached_rows["slot"] == slot) & (cached_rows["rank"] == 0)]
            if len(sel):
                ref = np.load(series_path(cache_dir, "train", study, getattr(sel.iloc[0], SERIES_COL)), mmap_mode="r")
                ok = vol.shape == ref.shape and bool((vol == np.asarray(ref)).all())
                per_series.append({ID_COL: study, SERIES_COL: f"slot:{slot}", "shape_ok": vol.shape == ref.shape,
                                   "equal_frac": float((vol == np.asarray(ref)).mean()) if vol.shape == ref.shape else 0.0,
                                   "max_abs_diff": 0 if ok else -1})
    df = pd.DataFrame(per_series)
    weighted = float(np.average(df["equal_frac"], weights=[1.0] * len(df))) if len(df) else float("nan")
    rep = {"n_studies": len(picks), "n_series": int(len(df)), "n_series_bit_exact": n_equal,
           "equal_frac_mean": weighted, "equal_frac_min": float(df["equal_frac"].min()),
           "all_shapes_ok": bool(df["shape_ok"].all()), "seconds": time.monotonic() - t0,
           "worst": df.sort_values("equal_frac").head(5).to_dict("records"),
           "passed": bool(df["shape_ok"].all() and weighted >= 0.999)}
    return rep


def precision(args) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from rsna_kad.cache import get_index
    from rsna_kad.config import load_config
    from rsna_kad.constants import ID_COL, LABEL_KEYS
    from rsna_kad.cv import load_folds
    from rsna_kad.data import StudyDataset, load_targets
    from rsna_kad.metrics import macro_auc
    from rsna_kad.model import load_bundle
    from rsna_kad.train import PRED_COLS, cache_root, predict_studies_cached, run_dir_for, targets_path

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = run_dir_for(cfg, args.fold, args.runs_dir)
    model, _ = load_bundle(run_dir, device)
    folds = load_folds(cfg.folds_csv)
    cached = set(get_index(cache_root(cfg)).studies("train"))
    targets = load_targets(targets_path(cfg, args.fold))
    ids = sorted(s for s in folds.loc[(folds["fold"] == args.fold) & ~folds["is_gold"], ID_COL].astype(str)
                 if s in cached and s in set(targets[ID_COL]))
    rng = np.random.default_rng(args.seed)
    ids = sorted(rng.choice(ids, size=min(args.n, len(ids)), replace=False))
    ds = StudyDataset(cache_root(cfg), ids, None, cfg, train=False)
    out = {"fold": args.fold, "n_studies": len(ids), "device": device}
    preds = {}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        if dtype == torch.float16 and device == "cpu":
            continue
        t0 = time.monotonic()
        pred = predict_studies_cached(model, ds, tta=False, batch_size=cfg.eval_batch_studies,
                                      num_workers=args.workers, device=device, dtype=dtype)
        el = time.monotonic() - t0
        merged = targets.merge(pred, on=ID_COL)
        y = merged[[f"{k}_target" for k in LABEL_KEYS]].to_numpy()
        w = merged[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy()
        macro, per_label = macro_auc(y, merged[PRED_COLS].to_numpy(), w)
        preds[name] = pred.set_index(ID_COL)[PRED_COLS].to_numpy()
        out[name] = {"macro_auc": macro, "seconds": el, "per_label": per_label}
    if "fp16" in preds:
        diff = np.abs(preds["fp16"] - preds["fp32"])
        out["auc_gap"] = abs(out["fp16"]["macro_auc"] - out["fp32"]["macro_auc"])
        out["max_abs_prob_diff"] = float(diff.max())
        out["mean_abs_prob_diff"] = float(diff.mean())
        out["passed"] = bool(out["auc_gap"] < 0.003)
    return out


def kinfer(args) -> dict:
    """OOF macro AUC of a finished run at several K_infer values, on the cached studies.

    The runs are trained and scored at the config's K_infer; the Kaggle kernel may run a smaller
    K to save time (§2.7). This measures what that costs on exactly the rows and targets the run
    was accepted on, without touching runs/<run_id>/oof.parquet.
    """
    import copy

    import pandas as pd
    import torch
    from rsna_kad.cache import get_index
    from rsna_kad.config import load_config
    from rsna_kad.constants import ID_COL
    from rsna_kad.cv import load_folds
    from rsna_kad.data import StudyDataset, load_targets
    from rsna_kad.model import load_bundle
    from rsna_kad.train import (PRED_COLS, amp_dtype, cache_root, evaluate, gold_frame_or_none, labels_path,
                                predict_studies_cached, run_dir_for, targets_path)

    base = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = load_folds(base.folds_csv)
    folds[ID_COL] = folds[ID_COL].astype(str)
    cached = set(get_index(cache_root(base)).studies("train"))
    gold = gold_frame_or_none()
    lp = labels_path(base)
    labels = pd.read_parquet(lp) if lp else None
    targets = pd.concat([load_targets(targets_path(base, k)).assign(fold=k) for k in args.folds], ignore_index=True)
    targets[ID_COL] = targets[ID_COL].astype(str)
    fold_of = dict(zip(folds[ID_COL], folds["fold"]))
    targets = targets[[fold_of.get(s, -1) == f for s, f in zip(targets[ID_COL], targets["fold"])]]

    out = {"run_id": base.run_id, "device": device, "k_values": list(args.k), "by_k": {}}
    for k in args.k:
        cfg = copy.deepcopy(base)
        cfg.model.k_infer = int(k)
        parts, t0 = [], time.monotonic()
        for fold in args.folds:
            run_dir = run_dir_for(cfg, fold, args.runs_dir)
            model, _ = load_bundle(run_dir, device, views=True)
            ids = sorted(s for s in folds.loc[folds["fold"] == fold, ID_COL] if s in cached)
            ds = StudyDataset(cache_root(cfg), ids, None, cfg, train=False)
            pred = predict_studies_cached(model, ds, tta=False, batch_size=args.batch_studies,
                                          num_workers=args.workers, device=device, dtype=amp_dtype(cfg))
            parts.append(pred)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        oof = pd.concat(parts, ignore_index=True)
        m = evaluate(oof[[ID_COL, *PRED_COLS]], targets.drop(columns=["fold"]), gold, labels, n_boot=args.n_boot)
        m["seconds"] = time.monotonic() - t0
        m["n_rows"] = int(len(oof))
        if device == "cuda":
            m["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["by_k"][str(k)] = m
    ref = str(max(args.k))
    for k, m in out["by_k"].items():
        m["delta_weak_vs_ref"] = m["oof_weak_auc"] - out["by_k"][ref]["oof_weak_auc"]
        m["delta_gold_vs_ref"] = m["gold_auc"] - out["by_k"][ref]["gold_auc"]
    out["reference_k"] = int(ref)
    out["tolerance"] = args.tolerance
    out["passed"] = all(abs(m["delta_weak_vs_ref"]) <= args.tolerance for m in out["by_k"].values())
    return out


def timing(args) -> dict:
    import pandas as pd
    from rsna_kad.constants import ID_COL
    from rsna_kad.infer import (members_from_ensemble, predict_studies, read_ensemble, test_study_dirs,
                                write_submission)

    data_dir = Path(args.data_dir)
    doc = read_ensemble(args.weights)
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype={ID_COL: str})
    series_df = pd.read_csv(data_dir / "test_series.csv", dtype=str)
    study_dirs = test_study_dirs(data_dir, series_df, sample)
    if args.replicate > 1:
        # The 3 public studies cannot fill the producer/consumer pipeline; replicas under new
        # ids give the steady-state rate that the 1,300-study projection needs.
        parts = []
        for i in range(args.replicate):
            part = series_df.copy()
            part[ID_COL] = part[ID_COL] + f"#{i}"
            parts.append(part)
        study_dirs = {f"{s}#{i}": d for i in range(args.replicate) for s, d in study_dirs.items()}
        series_df = pd.concat(parts, ignore_index=True)
    runs = []
    for i in range(args.repeat):
        members = members_from_ensemble(args.weights, doc)
        t0 = time.monotonic()
        pred = predict_studies(members, study_dirs, series_df, devices=args.device, workers=args.workers,
                               fp16=not args.fp32, prior=doc.get("prior"))
        total = time.monotonic() - t0
        runs.append({"total_s": total, "per_study_s": total / max(len(pred), 1),
                     "prep_s_mean": float(pred["prep_s"].mean()), "gpu_s_mean": float(pred["gpu_s"].mean()),
                     "errors": int((pred["error"] != "").sum())})
    if args.replicate == 1:
        write_submission(pred, data_dir / "sample_submission.csv", args.out_csv, prior=doc.get("prior"))
    best = min(r["per_study_s"] for r in runs)
    return {"weights": args.weights, "devices": args.device, "n_studies": len(pred), "members": len(doc["members"]),
            "replicate": args.replicate,
            "runs": runs, "per_study_s_best": best,
            "projected_1300_a6000_min": best * N_TEST_STUDIES / 60.0,
            "projected_1300_t4_min": best * N_TEST_STUDIES * T4_FACTOR / 60.0,
            "projected_1300_t4_with_margin_min": best * N_TEST_STUDIES * T4_FACTOR * 1.25 / 60.0,
            "submission": args.out_csv, "passed": bool(best < 60.0 and runs[-1]["errors"] == 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=os.environ.get("RSNA_KAD_DATA", "/raid/sroziewski/data/rsna_kad"))
    common.add_argument("--cache-dir",
                        default=os.environ.get("RSNA_KAD_CACHE", "/raid/sroziewski/cache/rsna_kad") + "/v1")
    common.add_argument("--out", default=None, help="write the JSON report here too")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parity", parents=[common])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--slots", nargs="*", default=None)
    p = sub.add_parser("precision", parents=[common])
    p.add_argument("--config", default="configs/mvp_effv2s_256.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--gpu", type=int, default=None)
    p = sub.add_parser("kinfer", parents=[common])
    p.add_argument("--config", required=True)
    p.add_argument("--k", type=int, nargs="+", default=[12, 24])
    p.add_argument("--folds", type=int, nargs="*", default=list(range(5)))
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--batch-studies", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=0.002)
    p.add_argument("--gpu", type=int, default=None)
    p = sub.add_parser("time", parents=[common])
    p.add_argument("--weights", required=True)
    p.add_argument("--device", action="append", default=None)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--replicate", type=int, default=1, help="run each public study this many times")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--out-csv", default="export/submission_bench.csv")
    args = ap.parse_args(argv)
    if getattr(args, "gpu", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    rep = {"parity": parity, "precision": precision, "kinfer": kinfer, "time": timing}[args.cmd](args)
    text = json.dumps(rep, indent=1, default=str)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if rep.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
