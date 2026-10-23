#!/usr/bin/env python3
"""data/targets/v3_pseudo/fold{k}.parquet = v1_1 targets mixed 0.5/0.5 with the main_v3 teacher.

  python qa/pseudo_v3/make_targets_v3.py            # build + validate + stats
  python qa/pseudo_v3/make_targets_v3.py --check    # validate an existing build only

Per fold k: rows of folds != k get target 0.5*v1_1 + 0.5*teacher on all 12 labels and the v1_1
weight rule re-run with absent_under_reported 0.1 -> 0.5; fold k's own rows (never trained on by
the fold-k student; they are its validation rows) stay bit-identical to v1_1. Gold is excluded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsna_kad.constants import ID_COL, LABEL_KEYS, UNDER_REPORTED_KEYS  # noqa: E402
from rsna_kad.cv import load_folds  # noqa: E402
from rsna_kad.data import load_targets  # noqa: E402
from rsna_kad.labels.pseudo import (V3_MIX, V3_SILENT_WEIGHT, mix_with_teacher,  # noqa: E402
                                    silent_weight_targets, teacher_mean)
from rsna_kad.labels.targets import Calibration, TargetsConfig, validate_targets  # noqa: E402

TEACHERS = ("runs/main_convnext_s_384", "runs/eff_effv2s_256", "runs/main_armc_convnextv2_t_384")
N_ROWS = 4349


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="data/targets/v1_1")
    ap.add_argument("--out", default="data/targets/v3_pseudo")
    ap.add_argument("--folds-csv", default="data/folds/folds_v1.csv")
    ap.add_argument("--teachers", nargs="+", default=list(TEACHERS),
                    help="run dirs whose OOF sigmoids are averaged (round 2: the round-1 students)")
    ap.add_argument("--version", default="v3_pseudo")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    base, out = ROOT / args.base, ROOT / args.out
    man = json.loads((base / "manifest.json").read_text())
    cfg = TargetsConfig.load(ROOT / man["config"])
    labels = pd.read_parquet(ROOT / man["labels"])
    labels[ID_COL] = labels[ID_COL].astype(str)
    folds = load_folds(ROOT / args.folds_csv)
    folds[ID_COL] = folds[ID_COL].astype(str)
    gold_ids = set(folds.loc[folds["is_gold"], ID_COL])
    oofs = [pd.read_parquet(ROOT / t / "oof.parquet") for t in args.teachers]
    for t, o in zip(args.teachers, oofs):
        assert len(o) == 4407 and o[ID_COL].is_unique, f"{t}: {len(o)} rows"
    teacher = teacher_mean(oofs)
    teacher = teacher[~teacher[ID_COL].isin(gold_ids)].reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    teacher.to_parquet(out / "teacher_oof_weak.parquet", index=False)

    stats = []
    for k in range(5):
        lt = load_targets(base / f"fold{k}.parquet")
        calib = Calibration.from_json(base / f"calibration_fold{k}.json")
        own = set(folds.loc[folds["fold"] == k, ID_COL])
        train_ids = set(lt[ID_COL]) - own
        if args.check:
            new = load_targets(out / f"fold{k}.parquet")
        else:
            rw = silent_weight_targets(labels, calib, cfg, exclude_ids=gold_ids)
            rw[ID_COL] = rw[ID_COL].astype(str)
            rw = rw.set_index(ID_COL).loc[lt[ID_COL]].reset_index()
            new, cnt = mix_with_teacher(lt, teacher, rw, train_ids)
            assert cnt["no_teacher"] == 0, cnt
            new.to_parquet(out / f"fold{k}.parquet", index=False)
            (out / f"calibration_fold{k}.json").write_bytes((base / f"calibration_fold{k}.json").read_bytes())
        # contract + leakage checks
        validate_targets(new, n_rows=N_ROWS)
        assert not set(new[ID_COL]) & gold_ids, "gold row in targets"
        assert list(new[ID_COL]) == list(lt[ID_COL])
        vo = new[ID_COL].isin(own).to_numpy()
        for c in [f"{x}_{f}" for x in LABEL_KEYS for f in ("target", "weight")]:
            assert np.array_equal(new.loc[vo, c].to_numpy(), lt.loc[vo, c].to_numpy()), f"fold {k} own rows moved: {c}"
        tr = ~vo
        tv = teacher.set_index(ID_COL).loc[new.loc[tr, ID_COL]]
        for x in LABEL_KEYS:
            want = (1 - V3_MIX) * lt.loc[tr, f"{x}_target"].to_numpy(np.float64) + V3_MIX * tv[f"{x}_pred"].to_numpy(np.float64)
            assert np.allclose(new.loc[tr, f"{x}_target"].to_numpy(np.float64), want, atol=1e-6), x
            w0, w1 = lt.loc[tr, f"{x}_weight"].to_numpy(), new.loc[tr, f"{x}_weight"].to_numpy()
            if x not in UNDER_REPORTED_KEYS:
                assert np.array_equal(w0, w1), f"{x}: well-reported weight moved"
            else:
                assert (w1 >= w0 - 1e-7).all(), f"{x}: a weight fell"
            stats.append({"fold": k, "key": x, "n_train": int(tr.sum()),
                          "t_v1_1": float(lt.loc[tr, f"{x}_target"].mean()),
                          "t_v3": float(new.loc[tr, f"{x}_target"].mean()),
                          "mean_abs_move": float(np.abs(new.loc[tr, f"{x}_target"].to_numpy() - lt.loc[tr, f"{x}_target"].to_numpy()).mean()),
                          "w_v1_1": float(w0.mean()), "w_v3": float(w1.mean()),
                          "n_w_raised": int((w1 > w0 + 1e-7).sum()),
                          "silent_t_v1_1": float(lt.loc[tr, f"{x}_target"][w1 > w0 + 1e-7].mean()) if (w1 > w0 + 1e-7).any() else float("nan"),
                          "silent_t_v3": float(new.loc[tr, f"{x}_target"][w1 > w0 + 1e-7].mean()) if (w1 > w0 + 1e-7).any() else float("nan"),
                          "teacher_mean": float(tv[f"{x}_pred"].mean()),
                          "pos_v1_1": float((lt.loc[tr, f"{x}_target"] >= 0.6).mean()),
                          "pos_v3": float((new.loc[tr, f"{x}_target"] >= 0.6).mean())})
        print(f"fold {k}: OK {len(new)} rows, {int(tr.sum())} mixed, {int(vo.sum())} own rows bit-identical to v1_1")
    s = pd.DataFrame(stats)
    s.to_csv(ROOT / f"qa/pseudo_v3/target_stats_by_fold{'' if args.version == 'v3_pseudo' else '_' + args.version}.csv", index=False)
    agg = s.groupby("key", sort=False).mean(numeric_only=True).drop(columns="fold")
    pd.set_option("display.width", 250)
    print(agg.round(3).to_string())
    if not args.check:
        (out / "manifest.json").write_text(json.dumps({
            "version": args.version, "base": args.base, "labels": man["labels"], "config": man["config"],
            "folds": args.folds_csv, "teachers": list(args.teachers), "mix": V3_MIX,
            "absent_under_reported_weight": V3_SILENT_WEIGHT, "n_rows": N_ROWS,
            "builder": "qa/pseudo_v3/make_targets_v3.py"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
