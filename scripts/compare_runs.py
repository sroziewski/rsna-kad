#!/usr/bin/env python3
"""OOF comparison table across runs, on identical rows and identical targets (§2.5 metrics).

  python scripts/compare_runs.py --runs runs/mvp_effv2s_256 runs/main_convnext_s_384 --fold 0
  python scripts/compare_runs.py --runs runs/mvp_effv2s_256 runs/main_convnext_s_384 --targets v1 --out qa/stage6/compare.md

Each run's OOF is read from runs/<run_id>/oof.parquet (5-fold assembly) or, with --fold, from
runs/<run_id>/fold<k>/oof.parquet. Rows are restricted to the studies every run predicted, so
the numbers are paired. OOF-weak/strict are recomputed on --targets (default: each run's own
targets_version, which makes "weak" incomparable when the versions differ - the table says
which table was used); gold-58 uses train.csv. Epoch information (selected/best/last) comes
from the per-fold metrics.json when present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (runs/<run_id>)")
    ap.add_argument("--fold", type=int, default=None, help="compare one fold instead of the assembled OOF")
    ap.add_argument("--targets", default=None, help="targets version to score every run on (e.g. v1)")
    ap.add_argument("--targets-dir", default="data/targets")
    ap.add_argument("--out", default=None, help="write the markdown report here as well")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    from rsna_kad.config import load_config  # noqa: E402
    from rsna_kad.constants import ID_COL, LABEL_KEYS  # noqa: E402
    from rsna_kad.cv import load_folds  # noqa: E402
    from rsna_kad.data import load_targets  # noqa: E402
    from rsna_kad.train import (METRICS_NAME, OOF_NAME, PRED_COLS, evaluate, gold_frame_or_none,  # noqa: E402
                                labels_path, targets_path)

    runs = []
    for r in args.runs:
        root = Path(r)
        cfg = load_config(root / "config.yaml" if (root / "config.yaml").exists() else root / "fold0" / "config.yaml")
        oof_path = root / (f"fold{args.fold}" if args.fold is not None else "") / OOF_NAME
        if not oof_path.exists():
            print(f"{root}: no {oof_path}, skipped", file=sys.stderr)
            continue
        oof = pd.read_parquet(oof_path)
        oof[ID_COL] = oof[ID_COL].astype(str)
        met_path = oof_path.with_name(METRICS_NAME)
        met = json.loads(met_path.read_text(encoding="utf-8")) if met_path.exists() else {}
        runs.append({"root": root, "cfg": cfg, "oof": oof, "metrics": met})
    if not runs:
        raise SystemExit("nothing to compare")

    common = set.intersection(*(set(r["oof"][ID_COL]) for r in runs))
    folds_df = load_folds(runs[0]["cfg"].folds_csv)
    folds_df[ID_COL] = folds_df[ID_COL].astype(str)
    if args.fold is not None:
        common &= set(folds_df.loc[folds_df["fold"] == args.fold, ID_COL])
    gold = gold_frame_or_none()
    gold_ids = set(folds_df.loc[folds_df["is_gold"], ID_COL])

    def targets_for(cfg) -> tuple[pd.DataFrame, str]:
        """Per-fold target tables (each fold's rows scored against its own calibration)."""
        version = args.targets or cfg.targets_version
        parts = []
        for k in sorted(folds_df["fold"].unique()):
            p = Path(args.targets_dir) / version / f"fold{k}.parquet"
            t = load_targets(p)
            ids = set(folds_df.loc[folds_df["fold"] == k, ID_COL])
            parts.append(t[t[ID_COL].isin(ids)])
        return pd.concat(parts, ignore_index=True), version

    labels_cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for r in runs:
        cfg = r["cfg"]
        oof = r["oof"][r["oof"][ID_COL].isin(common)].reset_index(drop=True)
        targets, version = targets_for(cfg)
        lp = labels_path(cfg)
        key = str(lp)
        if key not in labels_cache:
            labels_cache[key] = pd.read_parquet(lp) if lp else None
        m = evaluate(oof, targets, gold, labels_cache[key], n_boot=args.n_boot)
        met = r["metrics"]
        rows.append({
            "run": cfg.run_id, "backbone": cfg.model.backbone, "px": cfg.model.img_px,
            "slots": len(cfg.model.slots), "epochs": cfg.epochs, "targets": version,
            "trained_on": cfg.targets_version, "n_rows": len(oof), "n_weak": m["n_weak"], "n_gold": m["n_gold"],
            "weak": m["oof_weak_auc"], "strict": m["oof_strict_auc"], "gold": m["gold_auc"],
            "gold_ci": m["gold_ci95"], "per_label": m["per_label"],
            "per_label_gold": m.get("per_label_gold", {}),
            "policy": met.get("select_epoch", "best" if met else ""),
            "epoch": met.get("epoch"), "best_epoch": (met.get("best_epoch") or {}).get("epoch", met.get("epoch")),
            "best_weak": (met.get("best_epoch") or {}).get("oof_weak_auc", met.get("oof_weak_auc")),
            "last_weak": (met.get("last_epoch") or {}).get("oof_weak_auc",
                                                          (met.get("history") or [{}])[-1].get("oof_weak_auc")),
            "history": met.get("history") or [],
        })

    def f(x, nd=4) -> str:
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

    scope = f"fold {args.fold}" if args.fold is not None else "assembled OOF"
    lines = [f"# OOF comparison ({scope}, {len(common)} common studies, "
             f"{len(common & gold_ids)} gold)", "",
             "| run | backbone | px | slots | ep | trained on | scored on | policy | epoch (best/last) | "
             "OOF-weak | strict | gold | gold CI95 | n_weak | n_gold |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ep = f"{r['epoch']} ({r['best_epoch']}: {f(r['best_weak'])} / {f(r['last_weak'])})" if r["epoch"] is not None else "-"
        lines.append(f"| {r['run']} | {r['backbone']} | {r['px']} | {r['slots']} | {r['epochs']} | {r['trained_on']} | "
                     f"{r['targets']} | {r['policy']} | {ep} | {f(r['weak'])} | {f(r['strict'])} | {f(r['gold'])} | "
                     f"[{f(r['gold_ci'][0], 3)}, {f(r['gold_ci'][1], 3)}] | {r['n_weak']} | {r['n_gold']} |")
    lines += ["", "## Per-label OOF-weak AUC", "",
              "| label | " + " | ".join(r["run"] for r in rows) + (" | delta |" if len(rows) == 2 else " |"),
              "|---|" + "---|" * len(rows) + ("---|" if len(rows) == 2 else "")]
    for k in LABEL_KEYS:
        vals = [r["per_label"].get(k, float("nan")) for r in rows]
        delta = f" {f(vals[1] - vals[0])} |" if len(rows) == 2 else ""
        lines.append(f"| {k} | " + " | ".join(f(v) for v in vals) + " |" + delta)
    if any(r["per_label_gold"] for r in rows):
        lines += ["", "## Per-label gold AUC", "", "| label | " + " | ".join(r["run"] for r in rows) + " |",
                  "|---|" + "---|" * len(rows)]
        for k in LABEL_KEYS:
            lines.append(f"| {k} | " + " | ".join(f(r["per_label_gold"].get(k, float("nan"))) for r in rows) + " |")
    for r in rows:
        if r["history"]:
            lines += ["", f"## Epoch curve: {r['run']} ({scope})", "",
                      "| epoch | loss | weak | strict | gold | train s | img/s | peak GB |", "|---|---|---|---|---|---|---|---|"]
            for h in r["history"]:
                lines.append(f"| {h['epoch']} | {f(h.get('train_loss'))} | {f(h.get('oof_weak_auc'))} | "
                             f"{f(h.get('oof_strict_auc'))} | {f(h.get('gold_auc'))} | {f(h.get('train_s'), 0)} | "
                             f"{f(h.get('img_per_s'), 0)} | {f(h.get('peak_mem_gb'), 1)} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
