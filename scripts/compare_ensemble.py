#!/usr/bin/env python3
"""Paired comparison of ensemble combinations on the three metrics of architecture §2.5.

  python scripts/compare_ensemble.py --runs A=runs/main_convnext_s_384 E=runs/eff_effv2s_256 \
      C=runs/main_armc_convnextv2_t_384 --combos A C A+C A+E A+E+C [--folds 0] [--n-boot 1000]

scripts/compare_runs.py tabulates runs side by side; this one *blends* them (equal-weight mean
of sigmoids, §2.6) and answers "does adding this arm change anything". Every combination is
scored on the same studies, and each bootstrap replicate resamples those studies once and
scores every combination on that one resample, so the delta CIs are paired - without that,
the between-combination differences (~0.005) disappear inside each combination's own CI
(~0.006). The public LB cannot resolve differences below ~0.02 on ~390 studies, so this is
the evidence that decides whether an arm stays in the ensemble.

Also prints the per-label AUC table and the Spearman correlation between the arms' predictions.

Read-only: writes nothing.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna_kad.config import load_config  # noqa: E402
from rsna_kad.constants import ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.cv import load_folds  # noqa: E402
from rsna_kad.data import load_targets  # noqa: E402
from rsna_kad.metrics import auc, binarise  # noqa: E402
from rsna_kad.runguard import RunGuardError, assert_not_live  # noqa: E402
from rsna_kad.train import PRED_COLS, gold_frame_or_none, labels_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="NAME=path")
ap.add_argument("--combos", nargs="+", required=True, help="e.g. A A+E A+E+C")
ap.add_argument("--folds", type=int, nargs="*", default=None)
ap.add_argument("--targets", default="v1")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

runs = dict(r.split("=", 1) for r in a.runs)
try:
    assert_not_live(runs.values())
except RunGuardError as exc:
    sys.exit(f"{exc}")
preds: dict[str, pd.DataFrame] = {}
for name, root in runs.items():
    root = Path(root)
    parts = []
    if a.folds is None:
        p = root / "oof.parquet"
        parts = [pd.read_parquet(p)] if p.exists() else [
            pd.read_parquet(q) for q in sorted(root.glob("fold*/oof.parquet"))]
    else:
        parts = [pd.read_parquet(root / f"fold{k}" / "oof.parquet") for k in a.folds]
    df = pd.concat(parts, ignore_index=True)
    df[ID_COL] = df[ID_COL].astype(str)
    preds[name] = df.groupby(ID_COL, as_index=False)[PRED_COLS].mean()

cfg = load_config(Path(runs[next(iter(runs))]) / "fold0" / "config.yaml")
folds_df = load_folds(cfg.folds_csv)
folds_df[ID_COL] = folds_df[ID_COL].astype(str)
common = sorted(set.intersection(*(set(d[ID_COL]) for d in preds.values())))
if a.folds is not None:
    common = sorted(set(common) & set(folds_df.loc[folds_df["fold"].isin(a.folds), ID_COL]))
preds = {k: v[v[ID_COL].isin(common)].sort_values(ID_COL).reset_index(drop=True) for k, v in preds.items()}
ids = preds[next(iter(preds))][ID_COL].tolist()

targets = pd.concat([load_targets(Path("data/targets") / a.targets / f"fold{k}.parquet").pipe(
    lambda t, k=k: t[t[ID_COL].astype(str).isin(set(folds_df.loc[folds_df["fold"] == k, ID_COL]))])
    for k in sorted(folds_df["fold"].unique())], ignore_index=True)
targets[ID_COL] = targets[ID_COL].astype(str)
targets = targets.set_index(ID_COL).reindex(ids)
weak_rows = targets.notna().all(axis=1).to_numpy()
y_soft = targets[[f"{k}_target" for k in LABEL_KEYS]].to_numpy(dtype=np.float64)
w_soft = targets[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy(dtype=np.float64)

lp = labels_path(cfg)
labels = pd.read_parquet(lp).assign(**{ID_COL: lambda d: d[ID_COL].astype(str)}).set_index(ID_COL).reindex(ids) \
    if lp else None
gold = gold_frame_or_none()
gold[ID_COL] = gold[ID_COL].astype(str)
gold_pos = [i for i, s in enumerate(ids) if s in set(gold[ID_COL])]
gold_y = gold.set_index(ID_COL).reindex([ids[i] for i in gold_pos])[list(LABEL_KEYS)].to_numpy(dtype=np.float64)

# (rows, y_bin) per label for each metric; the same masks score every combination.
weak_masks, strict_masks = [], []
for j in range(len(LABEL_KEYS)):
    keep, y_bin = binarise(y_soft[:, j], w_soft[:, j])
    keep = keep & weak_rows
    weak_masks.append((np.flatnonzero(keep), y_bin[keep]))
    if labels is not None:
        st = labels[f"{LABEL_KEYS[j]}_state"].astype(str).to_numpy()
        # Gold rows are not in the targets table and are excluded here too, so "strict" means
        # the same rows as train.strict_auc (2.5 metric 2).
        k2 = ((st == "affirmed") | (st == "negated")) & weak_rows
        strict_masks.append((np.flatnonzero(k2), (st[k2] == "affirmed")))

combos = {c: [p.strip() for p in c.split("+")] for c in a.combos}
P = {c: np.mean([preds[m][PRED_COLS].to_numpy(dtype=np.float64) for m in ms], axis=0) for c, ms in combos.items()}


def macro(p: np.ndarray, masks) -> tuple[float, dict]:
    """Macro AUC over the labels that keep both classes."""
    per = {LABEL_KEYS[j]: auc(y, p[rows, j]) for j, (rows, y) in enumerate(masks)}
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def macro_on(p: np.ndarray, masks, take: np.ndarray) -> float:
    """Macro AUC on a bootstrap sample `take` (row indices, with repeats)."""
    per = []
    for j, (rows, y) in enumerate(masks):
        keep = np.isin(take, rows)
        if not keep.any():
            continue
        sample = take[keep]
        pos = np.searchsorted(rows, sample)
        v = auc(y[pos], p[sample, j])
        if not np.isnan(v):
            per.append(v)
    return float(np.mean(per)) if per else float("nan")


def gold_macro(p: np.ndarray, take: np.ndarray | None = None) -> float:
    rows = np.array(gold_pos) if take is None else take
    yy = gold_y if take is None else gold_y[np.searchsorted(np.array(gold_pos), take)]
    per = [auc(yy[:, j], p[rows, j]) for j in range(len(LABEL_KEYS))]
    per = [v for v in per if not np.isnan(v)]
    return float(np.mean(per)) if per else float("nan")


point, per_label = {}, {}
for c, p in P.items():
    w, pl = macro(p, weak_masks)
    s, _ = macro(p, strict_masks) if labels is not None else (float("nan"), {})
    point[c] = {"weak": w, "strict": s, "gold": gold_macro(p)}
    per_label[c] = pl

rng = np.random.default_rng(a.seed)
n = len(ids)
boot = {c: {"weak": [], "strict": [], "gold": []} for c in P}
gp = np.array(gold_pos)
for _ in range(a.n_boot):
    take = np.sort(rng.integers(0, n, n))
    gtake = np.sort(gp[rng.integers(0, len(gp), len(gp))])
    for c, p in P.items():
        boot[c]["weak"].append(macro_on(p, weak_masks, take))
        if labels is not None:
            boot[c]["strict"].append(macro_on(p, strict_masks, take))
        boot[c]["gold"].append(gold_macro(p, gtake))

names = list(P)
print(f"# ensemble comparison ({len(ids)} studies, {len(gold_pos)} gold, targets {a.targets}, "
      f"{a.n_boot} paired bootstrap replicates)\n")
print("| combination | OOF-weak [95% CI] | strict [95% CI] | gold-58 [95% CI] |")
print("|---|---|---|---|")
for c in names:
    cells = []
    for m in ("weak", "strict", "gold"):
        b = np.array([v for v in boot[c][m] if not np.isnan(v)])
        lo, hi = (np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan, np.nan)
        cells.append(f"{point[c][m]:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"| {c} | " + " | ".join(cells) + " |")

print("\n| delta | OOF-weak [95% CI] P(>0) | strict [95% CI] P(>0) | gold-58 [95% CI] P(>0) |")
print("|---|---|---|---|")
for i, c in enumerate(names):
    for d in names[i + 1:]:
        cells = []
        for m in ("weak", "strict", "gold"):
            x = np.array(boot[d][m]) - np.array(boot[c][m])
            x = x[~np.isnan(x)]
            if not len(x):
                cells.append("-")
                continue
            cells.append(f"{point[d][m] - point[c][m]:+.4f} [{np.percentile(x, 2.5):+.4f}, "
                         f"{np.percentile(x, 97.5):+.4f}] P={np.mean(x > 0):.3f}")
        print(f"| {d} - {c} | " + " | ".join(cells) + " |")

print("\n| label | " + " | ".join(names) + " |")
print("|---" * (len(names) + 1) + "|")
for k in LABEL_KEYS:
    print(f"| {k} | " + " | ".join(f"{per_label[c].get(k, float('nan')):.4f}" for c in names) + " |")

print("\n## arm diversity (Spearman correlation of predictions, mean over the 12 labels)\n")
print("| pair | rho |")
print("|---|---|")
arms = list(preds)
rank = {m: np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 0,
                               preds[m][PRED_COLS].to_numpy(dtype=np.float64)) for m in arms}
for i, x in enumerate(arms):
    for y in arms[i + 1:]:
        r = [np.corrcoef(rank[x][:, j], rank[y][:, j])[0, 1] for j in range(len(LABEL_KEYS))]
        print(f"| {x} vs {y} | {np.mean(r):.3f} |")
