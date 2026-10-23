#!/usr/bin/env python3
"""Stage 9 pseudo-labels for the under-reported keys, with a leakage-proof source split.

  python scripts/make_pseudo_targets.py sources          # inputs for the two source models
  python scripts/make_pseudo_targets.py build            # data/targets/v2/fold0.parquet

The honest experiment holds out fold 0. Its training set is folds 1-4, and every training row's
pseudo-label must come from a model that saw neither that row nor fold 0, so a fold-k OOF from
runs/main_convnext_s_384 cannot be used: those models were trained on four folds each and every
one of them saw fold 0. Two disjoint half-source models supply the labels instead:

  source A: trained on folds {1,2}  ->  labels folds {3,4}
  source B: trained on folds {3,4}  ->  labels folds {1,2}

Neither ever sees a fold-0 study, and neither labels a study it trained on.

`sources` writes, per source, a remapped folds CSV (pseudo-fold 0 = the folds it must predict,
pseudo-fold 1 = everything else) and a targets table holding only the rows it may train on -
train.py trains on `fold != k` intersected with the rows that have a target, so fold 0 is
excluded by the targets table and never enters either set. Both tables are a row subset of
data/targets/v1/fold0.parquet, which is calibrated on the gold rows of folds 1-4 only, so no
fold-0 gold reaches a source model either. Nothing under data/targets/v1 is modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = {"srcA": {"train": (1, 2), "label": (3, 4)},
       "srcB": {"train": (3, 4), "label": (1, 2)}}
EVAL_FOLD = 0
QA = Path("qa/stage9")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sources", "build"])
    ap.add_argument("--folds-csv", default="data/folds/folds_v1.csv")
    ap.add_argument("--v1", default="data/targets/v1/fold0.parquet")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="data/targets/v2")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import pandas as pd  # noqa: E402
    from rsna_kad.constants import ID_COL, LABEL_KEYS  # noqa: E402
    from rsna_kad.cv import load_folds, write_folds  # noqa: E402
    from rsna_kad.data import load_targets  # noqa: E402
    from rsna_kad.labels.pseudo import build_fold_targets  # noqa: E402

    folds = load_folds(args.folds_csv)
    folds[ID_COL] = folds[ID_COL].astype(str)
    v1 = load_targets(args.v1)

    if args.mode == "sources":
        QA.mkdir(parents=True, exist_ok=True)
        for name, spec in SRC.items():
            f = folds.copy()
            f["fold"] = (~f["fold"].isin(spec["label"])).astype(int)   # 0 = predict, 1 = the rest
            path = QA / f"folds_{name}.csv"
            md5 = write_folds(f, path)
            keep = set(folds.loc[folds["fold"].isin(spec["train"]), ID_COL])
            t = v1[v1[ID_COL].isin(keep)].reset_index(drop=True)
            tdir = QA / "targets" / name
            tdir.mkdir(parents=True, exist_ok=True)
            t.to_parquet(tdir / "fold0.parquet", index=False)
            assert not set(folds.loc[folds["fold"] == EVAL_FOLD, ID_COL]) & set(t[ID_COL]), \
                f"{name}: a fold-{EVAL_FOLD} study reached the source training table"
            print(f"{name}: train folds {spec['train']} ({len(t)} studies, no fold-{EVAL_FOLD} row), "
                  f"labels folds {spec['label']} ({int((f['fold'] == 0).sum())} rows); "
                  f"{path} md5 {md5}")
        return 0

    parts = []
    for name, spec in SRC.items():
        p = Path(args.runs_dir) / f"pseudo_{name}" / "fold0" / "oof.parquet"
        if not p.exists():
            raise SystemExit(f"missing {p} - train the source models first (see qa/stage9/report.md)")
        o = pd.read_parquet(p)
        o[ID_COL] = o[ID_COL].astype(str)
        want = set(folds.loc[folds["fold"].isin(spec["label"]), ID_COL])
        got = set(o[ID_COL])
        extra = got - want
        if extra:
            raise SystemExit(f"{name} predicted {len(extra)} studies outside folds {spec['label']}")
        print(f"{name}: {len(got)} predictions, covers {100.0 * len(got & want) / len(want):.1f} % "
              f"of folds {spec['label']}")
        parts.append(o[[ID_COL] + [f"{k}_pred" for k in LABEL_KEYS]])
    pred = pd.concat(parts, ignore_index=True)
    if pred[ID_COL].duplicated().any():
        raise SystemExit("the two sources predicted the same study - the split is not disjoint")
    eval_ids = set(folds.loc[folds["fold"] == EVAL_FOLD, ID_COL])
    if eval_ids & set(pred[ID_COL]):
        raise SystemExit(f"a fold-{EVAL_FOLD} study has a pseudo-label")

    out, counts = build_fold_targets(EVAL_FOLD, v1, folds, pred, args.out)
    held = out[ID_COL].isin(eval_ids)
    same = all((out.loc[held, c].to_numpy() == v1.loc[held.to_numpy(), c].to_numpy()).all()
               for k in LABEL_KEYS for c in (f"{k}_target", f"{k}_weight"))
    print(f"\n{args.out}/fold{EVAL_FOLD}.parquet: {len(out)} rows, "
          f"{sum(counts.values())} cells pseudo-labelled")
    for k, v in counts.items():
        print(f"  {k:10s} {v:5d}")
    print(f"fold-{EVAL_FOLD} rows bit-identical to v1: {same}")
    if not same:
        raise SystemExit("the held-out fold's rows changed - refusing to ship this table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
