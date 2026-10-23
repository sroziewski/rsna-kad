#!/usr/bin/env python3
"""Score any set of runs with every offline arbiter and print the comparison table.

  python scripts/evaluate_runs.py --runs runs/main_convnext_s_384 runs/eff_effv2s_256 \
      --blend --n-boot 2000 --out qa/arbiter/arbiters.md

Each arbiter is a `metrics.Subset` - which (study, label) cells count and what their binary
label is - and every candidate is scored on exactly those cells, so the arbiters differ only
in the cells they trust, never in the AUC computation. For each arbiter the table reports its
cell coverage, how well its own labels match the 58 image-derived gold rows, the bootstrap CI
of the level, and the paired-bootstrap CI of the delta between candidates (the number that
decides what the arbiter can resolve). See qa/arbiter/report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x: float, nd: int = 4) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def build_arbiters(labels_rows, targets, gold_rows, include):
    """name -> Subset, built on whatever row set `labels_rows` / `targets` hold.

    The same builder is called twice: once on the weak rows (the scoring subsets) and once on
    the 58 gold rows (the fidelity subsets), so "how often is this arbiter's label right"
    is measured with exactly the rule that produces its score."""
    from rsna_kad.metrics import (consensus_subset, evidence_mask, gold_subset, strict_subset,
                                  weak_subset)

    ev = evidence_mask(labels_rows)
    if not np.array_equal(targets["StudyInstanceUID"].astype(str).to_numpy(),
                          labels_rows["StudyInstanceUID"].astype(str).to_numpy()):
        raise ValueError("targets and label rows are not aligned; sort both by StudyInstanceUID")
    subs = {"weak": weak_subset(targets), "strict": strict_subset(labels_rows),
            "consensus": consensus_subset(labels_rows),
            "consensus+absent": consensus_subset(labels_rows, include_agreed_absent=True)}
    subs["consensus_ev"] = subs["consensus"].restrict(ev, "consensus_ev")
    subs["weak_ev"] = subs["weak"].restrict(ev, "weak_ev")
    if gold_rows is not None:
        subs["gold58"] = gold_subset(gold_rows)
    return {k: v for k, v in subs.items() if not include or k in include}


def gold_targets(labels_gold, folds, cfg, targets_dir):
    """Calibrated targets for the gold rows, each row using its own fold's calibration - the
    calibration that never saw that row (§2.5). Only needed so the `weak` arbiter's label
    fidelity can be measured on gold at all; these targets never touch training."""
    from rsna_kad.labels.targets import Calibration, make_targets
    fold_of = dict(zip(folds["StudyInstanceUID"], folds["fold"].astype(int)))
    parts = []
    for k in sorted(set(fold_of.values())):
        ids = [s for s in labels_gold["StudyInstanceUID"] if fold_of.get(s) == k]
        if not ids:
            continue
        calib = Calibration.from_json(Path(targets_dir) / f"calibration_fold{k}.json")
        parts.append(make_targets(labels_gold[labels_gold["StudyInstanceUID"].isin(ids)], calib, cfg))
    return pd.concat(parts, ignore_index=True).sort_values("StudyInstanceUID").reset_index(drop=True)


def gold_fidelity(sub, gold):
    """How right the arbiter's OWN binary label is on the gold cells it keeps.

    `acc` is raw agreement and depends on the subset's class balance; `bal_acc` is the macro
    over labels of the AUC of the arbiter's label against gold, which for a binary predictor
    is its balanced accuracy - that is the number to compare arbiters with."""
    from rsna_kad.constants import LABEL_KEYS
    from rsna_kad.metrics import _rank_auc
    g = gold.set_index("StudyInstanceUID").loc[sub.ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
    if not sub.keep.any():
        return None
    per = [_rank_auc(g[sub.keep[:, j], j], sub.y[sub.keep[:, j], j].astype(np.float64))
           for j in range(len(LABEL_KEYS)) if sub.keep[:, j].any()]
    per = [v for v in per if np.isfinite(v)]
    return {"n": int(sub.keep.sum()), "acc": float((sub.y[sub.keep] == g[sub.keep]).mean()),
            "bal_acc": float(np.mean(per)) if per else float("nan"),
            "n_labels": len(per), "pos_rate": float(g[sub.keep].mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs holding oof.parquet")
    ap.add_argument("--blend", action="store_true", help="also score the equal-weight mean of all runs")
    ap.add_argument("--blend-of", nargs="*", default=None, help="run dir names to blend (default: all)")
    ap.add_argument("--arbiters", nargs="*", default=None, help="subset of arbiter names to run")
    ap.add_argument("--labels", default="data/labels/llm_v1.parquet")
    ap.add_argument("--targets-dir", default="data/targets/v1")
    ap.add_argument("--targets-config", default="configs/targets_v1.yaml")
    ap.add_argument("--folds-csv", default="data/folds/folds_v1.csv")
    ap.add_argument("--train-csv", default=None, help="train.csv with the 58 gold rows")
    ap.add_argument("--lb", nargs="*", default=[], metavar="RUN=SCORE",
                    help="observed public-LB scores, used only as a sanity filter on the ordering")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
    from rsna_kad.labels.targets import TargetsConfig, gold_frame  # noqa: E402
    from rsna_kad.metrics import (bootstrap_subset_auc, ci95, paired_delta, subset_auc,  # noqa: E402
                                  weighted_arbiter)
    from rsna_kad.runguard import RunGuardError, assert_not_live  # noqa: E402

    try:
        assert_not_live(args.runs)
    except RunGuardError as exc:
        print(exc, file=sys.stderr)
        return 2
    pcols = [f"{k}_pred" for k in LABEL_KEYS]
    preds: dict[str, pd.DataFrame] = {}
    for rd in args.runs:
        df = pd.read_parquet(Path(rd) / "oof.parquet")
        df[ID_COL] = df[ID_COL].astype(str)
        preds[Path(rd).name] = df.sort_values(ID_COL).reset_index(drop=True)
    if args.blend:
        members = args.blend_of or list(preds)
        ids = preds[members[0]][ID_COL]
        arr = np.mean([preds[m].set_index(ID_COL).loc[ids, pcols].to_numpy(np.float64) for m in members], axis=0)
        preds["mean(" + "+".join(m.split("_")[0] for m in members) + ")"] = pd.concat(
            [ids.reset_index(drop=True), pd.DataFrame(arr, columns=pcols)], axis=1)
    names = list(preds)

    labels = pd.read_parquet(args.labels)
    labels[ID_COL] = labels[ID_COL].astype(str)
    labels = labels.sort_values(ID_COL).reset_index(drop=True)
    folds = pd.read_csv(args.folds_csv)
    folds[ID_COL] = folds[ID_COL].astype(str)
    gold = gold_frame(args.train_csv or Path(DATA_DIR) / "train.csv")
    gold[ID_COL] = gold[ID_COL].astype(str)

    # Fold-local calibration (§2.5): each fold's rows are scored against that fold's targets.
    parts = []
    for k in sorted(folds["fold"].unique()):
        t = pd.read_parquet(Path(args.targets_dir) / f"fold{k}.parquet")
        t[ID_COL] = t[ID_COL].astype(str)
        parts.append(t[t[ID_COL].isin(set(folds.loc[folds["fold"] == k, ID_COL]))])
    targets = pd.concat(parts, ignore_index=True).sort_values(ID_COL).reset_index(drop=True)

    common = set.intersection(*(set(df[ID_COL]) for df in preds.values()))
    targets = targets[targets[ID_COL].isin(common)].reset_index(drop=True)
    gold = gold[gold[ID_COL].isin(common)].reset_index(drop=True)
    include = set(args.arbiters) if args.arbiters else None
    lab_weak = labels[labels[ID_COL].isin(set(targets[ID_COL]))].reset_index(drop=True)
    lab_gold = labels[labels[ID_COL].isin(set(gold[ID_COL]))].reset_index(drop=True)
    arbs = build_arbiters(lab_weak, targets, gold, include)
    cfg = TargetsConfig.load(args.targets_config)
    fid_subs = build_arbiters(lab_gold, gold_targets(lab_gold, folds, cfg, args.targets_dir), gold, include)

    lb = {}
    for spec in args.lb:
        n, v = spec.split("=")
        lb[n] = float(v)

    rows: dict[str, dict] = {}
    for tag, sub in arbs.items():
        aligned = {n: sub.align(preds[n]) for n in names}
        scores = {n: subset_auc(sub, aligned[n]) for n in names}
        boots = bootstrap_subset_auc(sub, aligned, n_boot=args.n_boot, seed=args.seed)
        best = max(names, key=lambda n: scores[n][0])
        pairs = {f"{a} - {b}": paired_delta(boots[a], boots[b]) for a, b in combinations(names, 2)}
        rows[tag] = {
            "coverage": sub.coverage,
            "labels_scored": int(sum(1 for v in scores[best][1].values() if np.isfinite(v))),
            "per_label_coverage": sub.per_label_coverage(),
            "gold_fidelity": gold_fidelity(fid_subs[tag], gold) if tag in fid_subs else None,
            "macro": {n: scores[n][0] for n in names},
            "per_label": {n: scores[n][1] for n in names},
            "ci": {n: list(ci95(boots[n])) for n in names},
            "best": best,
            "pairs": pairs,
            "rank": sorted(names, key=lambda n: -scores[n][0]),
        }

    # Composite (f). The weights are derived from two measured properties of each arbiter and
    # never from the leaderboard: w ~ (balanced label-vs-gold fidelity - 0.5) / half_width^2,
    # i.e. an arbiter counts in proportion to how right its labels are and to its precision.
    # On the 2026-09-21 tables that gives weak 0.58 / consensus 0.38 / gold58 0.04 - gold-58's
    # +-0.027 resolution reduces it to a rounding error, which is why it is used as a veto
    # (qa/arbiter/report.md) and not as a term in a score.
    W = {"weak": 0.58, "consensus": 0.38, "gold58": 0.04}
    composite = {n: weighted_arbiter({t: rows[t]["macro"][n] for t in rows}, W) for n in names}

    L: list[str] = ["# Offline arbiters", "",
                    f"{len(names)} candidates, {args.n_boot} bootstrap replicates (seed {args.seed}, "
                    "studies resampled once per arbiter and reused for every candidate, so deltas are paired).", "",
                    "## 1. What each arbiter trusts", "",
                    "| arbiter | cells kept | labels scored | gold cells | label == gold | balanced (label vs gold) | gold pos rate | CI95 width (best cand.) |",
                    "|---|---|---|---|---|---|---|---|"]
    for tag, r in rows.items():
        gf = r["gold_fidelity"]
        lo, hi = r["ci"][r["best"]]
        L.append(f"| {tag} | {r['coverage']:,} | {r['labels_scored']}/12 | "
                 f"{gf['n'] if gf else 0} | {_fmt(gf['acc'], 3) if gf else '-'} | "
                 f"{_fmt(gf['bal_acc'], 3) if gf else '-'} | {_fmt(gf['pos_rate'], 3) if gf else '-'} | "
                 f"{_fmt(hi - lo, 4)} |")

    L += ["", "## 2. Scores", "", "| candidate | " + " | ".join(rows) + " | composite |",
          "|---|" + "---|" * (len(rows) + 1)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(_fmt(rows[t]["macro"][n]) for t in rows) +
                 f" | {_fmt(composite[n])} |")
    if lb:
        L += ["", "Observed public LB (sanity filter only, ~390 studies => own noise ~+-0.01): "
              + ", ".join(f"{n} {v:.3f}" for n, v in lb.items()) + "."]

    L += ["", "## 3. Ranking per arbiter", "", "| arbiter | ordering (best first) |", "|---|---|"]
    for tag, r in rows.items():
        L.append(f"| {tag} | " + " > ".join(r["rank"]) + " |")
    if lb:
        order = sorted((n for n in names if n in lb), key=lambda n: -lb[n])
        L.append("| public LB | " + " > ".join(order) + " |")

    L += ["", "## 4. Paired deltas (what each arbiter can resolve)", "",
          "`half-width` is half the 95 % CI of the paired delta: an arbiter cannot resolve a "
          "difference smaller than that, however many metrics agree.", "",
          "| pair | " + " | ".join(f"{t} delta [CI] P" for t in rows) + " |",
          "|---|" + "---|" * len(rows)]
    for pair in list(rows.values())[0]["pairs"]:
        cells = []
        for t in rows:
            d = rows[t]["pairs"][pair]
            cells.append(f"{d['mean']:+.4f} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}] P={d['p_gt']:.2f}")
        L.append(f"| {pair} | " + " | ".join(cells) + " |")
    L += ["", "| arbiter | median paired half-width | narrowest | widest |", "|---|---|---|---|"]
    for tag, r in rows.items():
        h = [d["half_width"] for d in r["pairs"].values()]
        L.append(f"| {tag} | {np.median(h):.4f} | {np.min(h):.4f} | {np.max(h):.4f} |")

    L += ["", "## 5. Per-label coverage", "", "| label | " + " | ".join(rows) + " |",
          "|---|" + "---|" * len(rows)]
    for k in LABEL_KEYS:
        cells = []
        for t in rows:
            c = rows[t]["per_label_coverage"][k]
            cells.append(f"{c['n']:,} ({c['n_pos']}+/{c['n_neg']}-)")
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"arbiters": rows, "composite": composite, "weights": W, "lb": lb},
                       indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
