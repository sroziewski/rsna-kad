#!/usr/bin/env python3
"""Soft targets + weights per fold from a label table (architecture §2.1, §3.2).

  python scripts/make_targets.py --labels data/labels/llm_v1.parquet --config configs/targets_v1.yaml \
      --folds data/folds/folds_v1.csv --out data/targets/v1

--random-folds SEED assigns temporary random folds (acceptance test of stage 2 only, before
data/folds/folds_v1.csv exists); the output must be regenerated with the real folds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsna_kad.constants import DATA_DIR, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.targets import TargetsConfig, build_fold_targets, gold_frame  # noqa: E402


def random_folds(ids: pd.Series, seed: int, n_splits: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"StudyInstanceUID": ids.to_numpy(), "fold": rng.integers(0, n_splits, len(ids)),
                         "site_key": "random", "lang": "", "is_gold": False})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", default=f"{DATA_DIR}/train.csv")
    ap.add_argument("--folds", default="data/folds/folds_v1.csv")
    ap.add_argument("--random-folds", type=int, default=None, metavar="SEED")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = TargetsConfig.load(args.config)
    table = pd.read_parquet(args.labels)
    gold = gold_frame(args.train)
    if args.random_folds is not None:
        folds = random_folds(table["StudyInstanceUID"], args.random_folds)
        print(f"WARNING: temporary random folds (seed {args.random_folds}); regenerate with {args.folds}")
    else:
        if not Path(args.folds).exists():
            print(f"{args.folds} does not exist yet; use --random-folds for a temporary run", file=sys.stderr)
            return 2
        folds = pd.read_csv(args.folds)
        for col in ("StudyInstanceUID", "fold"):
            if col not in folds.columns:
                raise SystemExit(f"{args.folds} lacks column {col}")
    missing = set(table["StudyInstanceUID"]) - set(folds["StudyInstanceUID"])
    if missing:
        raise SystemExit(f"{len(missing)} label rows have no fold")

    result = build_fold_targets(table, folds, gold, cfg, args.out)
    for k, t in result.items():
        calib = json.loads((Path(args.out) / f"calibration_fold{k}.json").read_text())
        tv = t[[f"{key}_target" for key in LABEL_KEYS]].to_numpy()
        wv = t[[f"{key}_weight" for key in LABEL_KEYS]].to_numpy()
        print(f"fold {k}: {len(t)} rows, gold-train {calib['n_gold_train']}, target mean {tv.mean():.3f}, "
              f"weight mean {wv.mean():.3f}, pos(>=0.6) {float((tv >= 0.6).mean()):.3f}, neg(<=0.2) {float((tv <= 0.2).mean()):.3f}")
    manifest = {"version": cfg.version, "labels": args.labels, "config": args.config,
                "folds": "random" if args.random_folds is not None else args.folds,
                "random_seed": args.random_folds, "n_rows": int(len(next(iter(result.values()))))}
    (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
