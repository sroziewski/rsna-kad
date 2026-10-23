#!/usr/bin/env python3
"""Generate data/folds/folds_v1.csv once from analysis/out/study_summary.csv.

Refuses to overwrite an existing folds file (the md5 is pinned in tests/test_cv.py);
pass --force only when a new folds version is intended.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rsna_kad import cv
from rsna_kad.config import load_folds_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/folds_v1.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_folds_config(args.config)
    out = Path(cfg.out_csv)
    if out.exists() and not args.force:
        print(f"{out} exists (md5 {cv.md5_of(out)}); use --force to regenerate", file=sys.stderr)
        return 1
    folds = cv.make_folds(cfg.study_summary, cfg.n_splits, cfg.seed, cfg.group_col, cfg.strat_col,
                          cfg.method, cfg.n_trials)
    problems = cv.check_folds(folds, cfg.n_splits)
    print(cv.fold_summary(folds).to_string())
    print(f"rows={len(folds)} gold={int(folds['is_gold'].sum())} sites={folds['site_key'].nunique()}")
    if problems:
        print("acceptance problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    md5 = cv.write_folds(folds, out)
    print(f"wrote {out} md5={md5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
